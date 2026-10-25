use std::{
    env,
    fs::{File, read_to_string},
    io::{BufWriter, Write},
    path::PathBuf,
};

use anyhow::{Result, anyhow, bail};
#[cfg(feature = "unstable")]
use av1_grain::estimate_plane_noise;
use av1_grain::parse_grain_table;
use clap::{ArgGroup, Parser, Subcommand};
use dialoguer::Confirm;
use ffmpeg::{codec, format};
use grav1synth::{
    AnalyzeOptions, ApplyOptions, GrainSource, GrainTableSegment, PhotonNoiseOptions, analyze,
    apply, hevc, misc::get_frame_count,
    parser::{
        BitstreamParser,
        grain::{FilmGrainHeader, FilmGrainParams},
    },
    presets,
    reader::BitstreamReader,
    write_film_grain_segment,
};
use log::{error, info, warn};
use num_rational::Rational32;

#[allow(clippy::too_many_lines)]
#[allow(clippy::cognitive_complexity)]
pub fn main() -> Result<()> {
    if env::var("RUST_LOG").is_err() {
        // SAFETY: idk why this is even unsafe
        unsafe {
            env::set_var("RUST_LOG", "error,grav1synth=info");
        }
    }
    pretty_env_logger::init();

    let args = Args::parse();

    match args.command {
        Commands::Inspect {
            input,
            output,
            overwrite,
        } => {
            if input == output {
                error!(
                    "Input and output paths are the same. This is probably a typo, because this \
                     would overwrite your input. Exiting."
                );
                return Ok(());
            }

            if output.exists()
                && !overwrite
                && !Confirm::new()
                    .with_prompt(format!(
                        "File {} exists. Overwrite?",
                        output.to_string_lossy()
                    ))
                    .interact()?
            {
                warn!("Not overwriting existing file. Exiting.");
                return Ok(());
            }

            let reader = BitstreamReader::open(&input)?;
            if reader.get_video_stream()?.parameters().id() == codec::Id::HEVC {
                let grain_tables = hevc::inspect_hevc_grain_table(&input)?;
                if grain_tables.is_empty() {
                    info!(
                        "No x265 AFGS1 film grain SEI found--this video does not use supported HEVC grain synthesis"
                    );
                    return Ok(());
                }

                let mut output_file = BufWriter::new(File::create(&output)?);
                writeln!(&mut output_file, "filmgrn1")?;
                for segment in grain_tables {
                    write_film_grain_segment(&segment, &mut output_file)?;
                }
                output_file.flush()?;

                info!("Done, wrote grain table to {}", output.to_string_lossy());
                return Ok(());
            }
            let frame_rate = reader.get_video_details().frame_rate;
            let mut parser: BitstreamParser<false> = BitstreamParser::new(reader);
            let grain_headers = parser.get_grain_headers()?;

            if !grain_headers
                .iter()
                .any(|h| matches!(h, &FilmGrainHeader::UpdateGrain(_)))
            {
                info!("No film grain headers found--this video does not use grain synthesis");
                return Ok(());
            }

            // As you can expect, this may lead to odd behaviors with VFR.
            // VFR is cursed.
            let grain_tables = aggregate_grain_headers(grain_headers, frame_rate);

            let mut output_file = BufWriter::new(File::create(&output)?);
            writeln!(&mut output_file, "filmgrn1")?;
            for segment in grain_tables {
                write_film_grain_segment(&segment, &mut output_file)?;
            }
            output_file.flush()?;

            info!("Done, wrote grain table to {}", output.to_string_lossy());
        }
        Commands::Apply {
            input,
            output,
            overwrite,
            grain,
            preset,
            iso,
            chroma,
            replace,
        } => {
            let source = match (grain, preset, iso) {
                (Some(grain_path), None, None) => GrainSource::TableFile(grain_path),
                (None, Some(preset_name), None) => GrainSource::Preset(preset_name),
                (None, None, Some(iso_value)) => {
                    GrainSource::PhotonNoise(PhotonNoiseOptions { iso: iso_value, chroma })
                }
                // The ArgGroup on the Apply variant guarantees exactly one of grain/preset/iso
                // is Some, so this branch can never trigger at runtime.
                _ => unreachable!(
                    "clap ArgGroup enforces exactly one of --grain, --preset, or --iso"
                ),
            };
            apply(&ApplyOptions {
                input,
                output,
                overwrite,
                source,
                replace,
            })?;
        }
        Commands::Presets => {
            println!("Built-in film grain presets (use with `apply --preset <NAME>`):\n");

            println!("Available Presets:\n");
            for p in presets::STANDALONE_PRESETS {
                println!("  {}  ({})", p.name, p.description);
            }
            for fp in presets::FORMAT_PRESETS {
                println!("  {}  ({})", fp.name, fp.description);
            }

            println!();
            println!("Example: grav1synth apply input.mkv -o output.mkv --preset 16mm");

            println!();
            println!();
            // List the format preset names for the modifier section.
            let format_names = presets::FORMAT_PRESETS
                .iter()
                .map(|fp| fp.name)
                .collect::<Vec<_>>()
                .join(", ");
            println!("Available film stock modifiers (applies to {format_names}):\n");
            for stock in presets::FILM_STOCKS {
                if stock.suffix.is_empty() {
                    println!("       {}  (default)", stock.description);
                } else {
                    println!("  {}  {}", stock.suffix, stock.description);
                }
            }

            println!();
            println!("Example: grav1synth apply input.mkv -o output.mkv --preset 16mm-3");
        }
        Commands::Remove {
            input,
            output,
            overwrite,
        } => {
            if input == output {
                error!(
                    "Input and output paths are the same. This is probably a typo, because this \
                     would overwrite your input. Exiting."
                );
                return Ok(());
            }

            if output.exists()
                && !overwrite
                && !Confirm::new()
                    .with_prompt(format!(
                        "File {} exists. Overwrite?",
                        output.to_string_lossy()
                    ))
                    .interact()?
            {
                warn!("Not overwriting existing file. Exiting.");
                return Ok(());
            }

            let reader = BitstreamReader::open(&input)?;
            if reader.get_video_stream()?.parameters().id() == codec::Id::HEVC {
                hevc::modify_hevc_grain(&input, &output, None, true)?;
                info!("Done, wrote output file to {}", output.to_string_lossy());
                return Ok(());
            }
            let writer = format::output(&output)?;
            let mut parser: BitstreamParser<true> =
                BitstreamParser::with_writer(reader, writer, None);

            parser.modify_grain_headers()?;

            info!("Done, wrote output file to {}", output.to_string_lossy());
        }
        Commands::Diff {
            source,
            denoised,
            output,
            overwrite,
            filters,
            threads,
        } => {
            analyze(&AnalyzeOptions {
                source,
                denoised,
                output,
                overwrite,
                filters,
                threads,
            })?;
        }
        Commands::X265Aom {
            grain,
            output,
            overwrite,
            video,
            frames,
            fps,
        } => {
            if grain == output {
                error!(
                    "Input and output paths are the same. This is probably a typo, because this \
                     would overwrite your input. Exiting."
                );
                return Ok(());
            }

            if output.exists()
                && !overwrite
                && !Confirm::new()
                    .with_prompt(format!(
                        "File {} exists. Overwrite?",
                        output.to_string_lossy()
                    ))
                    .interact()?
            {
                warn!("Not overwriting existing file. Exiting.");
                return Ok(());
            }

            let grain_data = read_to_string(&grain)?;
            let grain_segments = parse_grain_table(&grain_data)?
                .into_iter()
                .map(GrainTableSegment::from)
                .collect::<Vec<_>>();

            if grain_segments.is_empty() {
                bail!("Grain table does not contain any enabled grain segments");
            }

            let (frame_count, frame_rate) = match (video, frames, fps) {
                (Some(video), None, None) => {
                    let frame_count = get_frame_count(&video)?;
                    let reader = BitstreamReader::open(&video)?;
                    (frame_count, Some(reader.get_video_details().frame_rate))
                }
                (None, Some(frame_count), fps) => {
                    let frame_rate = fps.as_deref().map(parse_fps).transpose()?;
                    (frame_count, frame_rate)
                }
                (Some(_), Some(_), _) => {
                    bail!("Use either --video or --frames, not both");
                }
                (Some(_), None, Some(_)) => {
                    bail!("--fps is only valid with --frames; --video already provides FPS");
                }
                (None, None, _) => {
                    bail!("Provide --video to infer frame count/FPS, or --frames manually");
                }
            };

            if frame_count == 0 {
                bail!("Frame count must be greater than zero");
            }
            if grain_segments.len() > 1 && frame_rate.is_none() {
                bail!("Multi-segment grain tables require --video or --fps to map timestamps");
            }

            let mut output_file = BufWriter::new(File::create(&output)?);
            write_x265_aom_file(&grain_segments, frame_count, frame_rate, &mut output_file)?;
            output_file.flush()?;

            info!(
                "Done, wrote {frame_count} x265 AOM grain record(s) to {}",
                output.to_string_lossy()
            );
        }
        #[cfg(feature = "unstable")]
        Commands::Estimate {
            source,
            output,
            overwrite,
            chroma,
        } => {
            if source == output {
                error!(
                    "Input and output paths are the same. This is probably a typo, because this \
                     would overwrite your input. Exiting."
                );
                return Ok(());
            }

            if output.exists()
                && !overwrite
                && !Confirm::new()
                    .with_prompt(format!(
                        "File {} exists. Overwrite?",
                        output.to_string_lossy()
                    ))
                    .interact()?
            {
                warn!("Not overwriting existing file. Exiting.");
                return Ok(());
            }

            let mut reader = BitstreamReader::open(&source)?;
            let bit_depth = reader.get_video_details().bit_depth;
            let mut frame_estimates = Vec::new();

            loop {
                match bit_depth {
                    8 => match reader.get_frame::<u8>()? {
                        Some(frame) => {
                            frame_estimates.push(estimate_plane_noise(&frame.y_plane, bit_depth));
                        }
                        None => {
                            break;
                        }
                    },
                    9..=16 => match reader.get_frame::<u16>()? {
                        Some(frame) => {
                            frame_estimates.push(estimate_plane_noise(&frame.y_plane, bit_depth));
                        }
                        None => {
                            break;
                        }
                    },
                    _ => {
                        bail!("Bit depths not between 8-16 are not currently supported");
                    }
                }
            }

            let video_stream = reader.get_video_stream().unwrap();
            // SAFETY: We immediately dereference the pointer to get the contained struct,
            // so there's no possibility of use-after-free later.
            let video_params = unsafe { *video_stream.parameters().as_ptr() };
            let frame_rate = reader.get_video_details().frame_rate;
            let trc = video_params.color_trc;

            let mut output_file = BufWriter::new(File::create(&output)?);
            writeln!(&mut output_file, "filmgrn1")?;
            for estimate in &frame_estimates {
                writeln!(&mut output_file, "{:.3}", estimate.unwrap_or(-1f64))?;
            }
            // for segment in build_segments_from_estimate(&frame_estimates, video_params,
            // frame_rate, chroma) {
            //     write_film_grain_segment(&segment.into(), &mut output_file)?;
            // }
            output_file.flush()?;
            info!("Done, wrote output file to {}", output.to_string_lossy());
        }
    }

    Ok(())
}



fn write_x265_aom_file<W: Write>(
    segments: &[GrainTableSegment],
    frame_count: usize,
    frame_rate: Option<Rational32>,
    output: &mut W,
) -> anyhow::Result<()> {
    if segments.len() == 1 {
        let params = &segments[0].grain_params;
        for _ in 0..frame_count {
            write_x265_aom_record(Some(params), output)?;
        }
        return Ok(());
    }

    let frame_rate = frame_rate.ok_or_else(|| anyhow!("Frame rate is required"))?;
    let time_per_frame =
        *frame_rate.denom() as f64 / *frame_rate.numer() as f64 * TIMESTAMP_BASE_UNIT;

    for frame in 0..frame_count {
        let timestamp = (frame as f64 * time_per_frame).floor() as u64;
        let params = segments
            .iter()
            .find(|segment| segment.start_time <= timestamp && timestamp < segment.end_time)
            .map(|segment| &segment.grain_params);
        write_x265_aom_record(params, output)?;
    }

    Ok(())
}

fn write_x265_aom_record<W: Write>(
    params: Option<&FilmGrainParams>,
    output: &mut W,
) -> anyhow::Result<()> {
    let Some(params) = params else {
        write_i32_le(output, 0)?; // apply_grain
        write_u16_le(output, 0)?; // grain_seed
        write_i32_le(output, 0)?; // update_grain
        write_i32_le(output, 0)?; // num_y_points
        write_i32_le(output, 0)?; // num_cb_points
        write_i32_le(output, 0)?; // num_cr_points
        write_i32_le(output, 0)?; // scaling_shift
        write_i32_le(output, 0)?; // ar_coeff_lag
        write_i32_le(output, 0)?; // ar_coeff_shift
        write_i32_le(output, 0)?; // grain_scale_shift
        write_i32_le(output, 0)?; // overlap_flag
        write_i32_le(output, 0)?; // clip_to_restricted_range
        return Ok(());
    };

    if params.chroma_scaling_from_luma {
        bail!(
            "x265's --aom-film-grain reader does not consume chroma_scaling_from_luma records; \
             generate or edit the grain table with explicit Cb/Cr scaling points"
        );
    }

    write_i32_le(output, 1)?; // apply_grain
    write_u16_le(output, params.grain_seed)?;
    write_i32_le(output, 1)?; // update_grain

    write_points(output, &params.scaling_points_y)?;
    write_points(output, &params.scaling_points_cb)?;
    write_points(output, &params.scaling_points_cr)?;

    write_i32_le(output, i32::from(params.scaling_shift))?;
    write_i32_le(output, i32::from(params.ar_coeff_lag))?;

    if !params.scaling_points_y.is_empty() {
        write_coeffs(output, params.ar_coeffs_y.iter().copied(), 24)?;
    }
    if !params.scaling_points_cb.is_empty() {
        write_coeffs(output, params.ar_coeffs_cb.iter().copied(), 25)?;
    }
    if !params.scaling_points_cr.is_empty() {
        write_coeffs(output, params.ar_coeffs_cr.iter().copied(), 25)?;
    }

    write_i32_le(output, i32::from(params.ar_coeff_shift))?;
    write_i32_le(output, i32::from(params.grain_scale_shift))?;

    if !params.scaling_points_cb.is_empty() {
        write_i32_le(output, i32::from(params.cb_mult))?;
        write_i32_le(output, i32::from(params.cb_luma_mult))?;
        write_i32_le(output, i32::from(params.cb_offset))?;
    }
    if !params.scaling_points_cr.is_empty() {
        write_i32_le(output, i32::from(params.cr_mult))?;
        write_i32_le(output, i32::from(params.cr_luma_mult))?;
        write_i32_le(output, i32::from(params.cr_offset))?;
    }

    write_i32_le(output, i32::from(params.overlap_flag))?;
    write_i32_le(output, i32::from(params.clip_to_restricted_range))?;

    Ok(())
}

fn write_points<W: Write>(output: &mut W, points: &[[u8; 2]]) -> anyhow::Result<()> {
    write_i32_le(output, points.len().try_into()?)?;
    for point in points {
        write_i32_le(output, i32::from(point[0]))?;
        write_i32_le(output, i32::from(point[1]))?;
    }
    Ok(())
}

fn write_coeffs<W: Write>(
    output: &mut W,
    coeffs: impl IntoIterator<Item = i8>,
    count: usize,
) -> anyhow::Result<()> {
    let mut written = 0usize;
    for coeff in coeffs.into_iter().take(count) {
        write_i32_le(output, i32::from(coeff))?;
        written += 1;
    }
    for _ in written..count {
        write_i32_le(output, 0)?;
    }
    Ok(())
}

fn write_i32_le<W: Write>(output: &mut W, value: i32) -> anyhow::Result<()> {
    output.write_all(&value.to_le_bytes())?;
    Ok(())
}

fn write_u16_le<W: Write>(output: &mut W, value: u16) -> anyhow::Result<()> {
    output.write_all(&value.to_le_bytes())?;
    Ok(())
}

fn parse_fps(value: &str) -> anyhow::Result<Rational32> {
    if let Some((numer, denom)) = value.split_once('/') {
        let numer = numer.parse::<i32>()?;
        let denom = denom.parse::<i32>()?;
        if numer <= 0 || denom <= 0 {
            bail!("FPS numerator and denominator must be positive");
        }
        return Ok(Rational32::new(numer, denom));
    }

    let fps = value.parse::<f64>()?;
    if !fps.is_finite() || fps <= 0.0 {
        bail!("FPS must be positive");
    }

    Ok(Rational32::new((fps * 1000.0).round() as i32, 1000))
}

// of a second.
const TIMESTAMP_BASE_UNIT: f64 = 10_000_000f64;

fn aggregate_grain_headers(
    grain_headers: &[FilmGrainHeader],
    frame_rate: Rational32,
) -> Vec<GrainTableSegment> {
    let time_per_packet: f64 =
        *frame_rate.denom() as f64 / *frame_rate.numer() as f64 * TIMESTAMP_BASE_UNIT;
    let mut cur_packet_start: u64 = 0;
    let mut cur_packet_end_f: f64 = time_per_packet;
    let mut cur_packet_end: u64 = cur_packet_end_f.ceil() as u64;

    grain_headers.iter().fold(Vec::new(), |mut acc, elem| {
        let prev_packet_has_grain = acc
            .last()
            .is_some_and(|last: &GrainTableSegment| last.end_time == cur_packet_start);
        if prev_packet_has_grain {
            match *elem {
                FilmGrainHeader::Disable => {
                    // Do nothing. This will disable film grain for this
                    // and future frames.
                }
                FilmGrainHeader::CopyRefFrame => {
                    // Increment the end time of the current table segment.
                    let cur_segment = acc.last_mut().expect("prev_packet_has_grain is true");
                    cur_segment.end_time = cur_packet_end;
                }
                FilmGrainHeader::UpdateGrain(ref grain_params) => {
                    let cur_segment = acc.last_mut().expect("prev_packet_has_grain is true");
                    if grain_params == &cur_segment.grain_params {
                        // Increment the end time of the current table segment.
                        cur_segment.end_time = cur_packet_end;
                    } else {
                        // The grain params changed, so we have to make a new segment.
                        acc.push(GrainTableSegment {
                            start_time: cur_packet_start,
                            end_time: cur_packet_end,
                            grain_params: grain_params.clone(),
                        });
                    }
                }
            }
        } else if let FilmGrainHeader::UpdateGrain(ref grain_params) = *elem {
            acc.push(GrainTableSegment {
                start_time: cur_packet_start,
                end_time: cur_packet_end,
                grain_params: grain_params.clone(),
            });
        }

        cur_packet_start = cur_packet_end;
        cur_packet_end_f += time_per_packet;
        cur_packet_end = cur_packet_end_f.ceil() as u64;
        acc
    })
}

#[derive(Parser, Debug)]
#[command(
    about = "Grain synth analyzer and editor for AV1 and HEVC files",
    version,
    flatten_help = true
)]
pub struct Args {
    #[clap(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Read the film grain table from an AV1 or HEVC video and write it to a file.
    ///
    /// Reports if the video has no film grain synthesis enabled.
    Inspect {
        /// The AV1 or HEVC file to inspect.
        #[clap(value_parser)]
        input: PathBuf,
        /// The path to write the film grain table to.
        #[clap(long, short, value_parser)]
        output: PathBuf,
        /// Overwrite the output file without prompting.
        #[clap(long, short = 'y')]
        overwrite: bool,
    },
    /// Applies film grain from a provided grain-table, a built-in preset, or generated
    /// photon-noise-based grain to a given AV1 or HEVC video and outputs it at a given `output` path.
    ///
    /// Exactly one grain source must be provided:
    ///   --grain <FILE>     apply grain from a table file
    ///   --preset <NAME>    apply a built-in film stock preset (see `grav1synth presets`)
    ///   --iso <NUM>        generate photon-noise-based grain (luma only; add --chroma for colour)
    ///
    /// If the input already has film grain headers the command skips by default.
    /// Pass --replace to overwrite existing grain instead.
    #[command(group(
        ArgGroup::new("grain_source")
            .required(true)
            .args(["grain", "preset", "iso"])
    ))]
    Apply {
        /// The AV1 or HEVC file to apply grain to.
        #[clap(value_parser)]
        input: PathBuf,
        /// The path to write the grain-synthed AV1 or HEVC file to.
        #[clap(long, short, value_parser)]
        output: PathBuf,
        /// Overwrite the output file without prompting.
        #[clap(long, short = 'y')]
        overwrite: bool,
        /// Path to a film grain table file.
        /// Cannot be used together with --preset or --iso.
        #[clap(long, short, value_parser)]
        grain: Option<PathBuf>,
        /// Name of a built-in film stock preset (case-insensitive).
        /// Run `grav1synth presets` to see all available names.
        /// Cannot be used together with --grain or --iso.
        #[clap(long, short)]
        preset: Option<String>,
        /// ISO strength for photon-noise-based grain (1–4294967295; 100–6400 recommended).
        /// Cannot be used together with --grain or --preset.
        #[clap(long, value_parser = clap::value_parser!(u32).range(1..))]
        iso: Option<u32>,
        /// Apply photon-noise grain to chroma planes as well as luma (only valid with --iso).
        #[clap(long, requires = "iso")]
        chroma: bool,
        /// Overwrite any existing grain headers in the input.
        /// Without this flag the command skips files that already have grain.
        #[clap(long)]
        replace: bool,
    },
    /// List all built-in film grain presets that can be used with `apply --preset`.
    Presets,
    /// Strip all supported film grain synthesis from an AV1 or HEVC video.
    Remove {
        /// The AV1 or HEVC file to remove grain from.
        #[clap(value_parser)]
        input: PathBuf,
        /// The path to write the grain-free AV1 file to.
        #[clap(long, short, value_parser)]
        output: PathBuf,
        /// Overwrite the output file without prompting.
        #[clap(long, short = 'y')]
        overwrite: bool,
    },
    /// Generate a film grain table by diffing a source video against a denoised copy.
    ///
    /// This produces the most accurate grain table because it measures the actual
    /// noise present in the source rather than estimating it.
    Diff {
        /// The untouched source file.
        #[clap(value_parser)]
        source: PathBuf,
        /// The denoised version of the source file.
        #[clap(value_parser)]
        denoised: PathBuf,
        /// The path to write the output film grain table to.
        #[clap(long, short, value_parser)]
        output: PathBuf,
        /// Overwrite the output file without prompting.
        #[clap(long, short = 'y')]
        overwrite: bool,
        /// A semicolon-separated list of filters to apply to the source before running the diff.
        /// For example: "crop:top=42,left=64;resize:width=1920,height=1080".
        ///
        /// Currently supported filters and options:
        /// - "crop": Crops the sides of the video
        ///   - Params: "top", "bottom", "left", "right"
        /// - "resize": Resizes the video
        ///   - Params: "width", "height", "alg"
        ///   - "alg" options are "hermite", "catmullrom", "mitchell", "lanczos", and "spline36"
        ///     Default is "catmullrom"
        #[clap(long, short, verbatim_doc_comment)]
        filters: Option<String>,
        /// Number of Rayon worker threads to use for grain analysis.
        /// Defaults to Rayon's normal thread count when omitted.
        #[clap(long)]
        threads: Option<usize>,
    },
    /// Convert a `filmgrn1` table into x265's binary `--aom-film-grain` input.
    ///
    /// x265 reads one binary grain record for every encoded frame, so this command
    /// expands timestamped grain-table segments to per-frame records. Provide
    /// --video to infer frame count and FPS from the encoded input, or use
    /// --frames and optionally --fps manually.
    X265Aom {
        /// The `filmgrn1` grain table to convert.
        #[clap(value_parser)]
        grain: PathBuf,
        /// The path to write the x265 AOM film-grain binary file to.
        #[clap(long, short, value_parser)]
        output: PathBuf,
        /// Overwrite the output file without prompting.
        #[clap(long, short = 'y')]
        overwrite: bool,
        /// Video whose frame count and FPS should be used for per-frame expansion.
        #[clap(long, value_parser)]
        video: Option<PathBuf>,
        /// Number of x265 grain records to write. Use when --video is unavailable.
        #[clap(long, value_parser)]
        frames: Option<usize>,
        /// Frame rate used to map multi-segment grain tables when --frames is used.
        /// Accepts ratios like 24000/1001 or decimal values like 23.976.
        #[clap(long, requires = "frames")]
        fps: Option<String>,
    },
    /// Analyzes a source video and estimates the amount of noise in the source,
    /// then generates an appropriate film grain table. This is less accurate
    /// than the diff method, but is significantly faster.
    #[cfg(feature = "unstable")]
    Estimate {
        /// The source file to inspect.
        #[clap(value_parser)]
        source: PathBuf,
        /// The path to the output film grain table.
        #[clap(long, short, value_parser)]
        output: PathBuf,
        /// Overwrite the output file without prompting.
        #[clap(long, short = 'y')]
        overwrite: bool,
        /// Whether to apply grain to the chroma planes as well.
        #[clap(long)]
        chroma: bool,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_fps_accepts_ratio() {
        let fps = parse_fps("24000/1001").expect("valid fps");
        assert_eq!(*fps.numer(), 24000);
        assert_eq!(*fps.denom(), 1001);
    }

    #[test]
    fn x265_aom_writer_expands_single_segment_per_frame() {
        let grain = r"filmgrn1
E 0 9223372036854775807 1 7391 1
	p 0 6 0 8 0 1 0 0 0 0 0 0
	sY 2 0 20 255 4
	sCb 0
	sCr 0
	cY
	cCb 0
	cCr 0
";
        let segments = parse_grain_table(grain)
            .expect("valid grain table")
            .into_iter()
            .map(GrainTableSegment::from)
            .collect::<Vec<_>>();
        let mut output = Vec::new();

        write_x265_aom_file(&segments, 2, None, &mut output).expect("write x265 records");

        assert_eq!(output.len(), 316);
        assert_eq!(&output[0..4], &1_i32.to_le_bytes());
        assert_eq!(&output[4..6], &7391_u16.to_le_bytes());
        assert_eq!(&output[6..10], &1_i32.to_le_bytes());
        assert_eq!(&output[10..14], &2_i32.to_le_bytes());
        assert_eq!(&output[158..162], &1_i32.to_le_bytes());
    }
}

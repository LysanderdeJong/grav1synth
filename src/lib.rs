//! Library interface for `grav1synth`: film-grain analysis and bitstream editing.
//!
//! The command-line binary in `main.rs` is a thin wrapper over [`analyze`] and
//! [`apply`]. Embedding applications (e.g. `shot-encoder grain`) should call
//! those entry points directly instead of shelling out to the binary.
//!
//! [`analyze`] diffs an untouched source video against a denoised copy and
//! writes a `filmgrn1` grain table. [`apply`] writes grain metadata from a
//! table file, a built-in preset, or generated photon noise into an AV1 or
//! HEVC bitstream without re-encoding.

pub mod filters;
pub mod hevc;
pub mod misc;
pub mod parser;
pub mod presets;
pub mod profile;
pub mod reader;

use std::{
    fs::{File, read_to_string},
    io::{BufWriter, Write, stderr},
    num::NonZeroU8,
    path::PathBuf,
    sync::mpsc::sync_channel,
    time::Duration,
};

use anyhow::{Result, anyhow, bail};
use av1_grain::{
    DiffGenerator, TransferFunction, generate_photon_noise_params, parse_grain_table,
    v_frame::{frame::Frame, pixel::Pixel},
};
use crossterm::tty::IsTty;
use ffmpeg::{
    codec,
    ffi::{AVColorRange, AVColorTransferCharacteristic},
    format,
};
use indicatif::{HumanDuration, ProgressBar, ProgressDrawTarget, ProgressState, ProgressStyle};
use log::{debug, info, warn};
use parser::grain::FilmGrainParams;
use rayon::ThreadPoolBuilder;
use v_frame_05::pixel::Pixel as Pixel05;

use crate::{
    filters::FilterChain, misc::get_frame_count, parser::BitstreamParser, reader::BitstreamReader,
};

/// A single timestamped run of identical grain parameters, as stored in a
/// `filmgrn1` grain table.
#[derive(Debug, Clone)]
pub struct GrainTableSegment {
    /// Start of the segment, in 100ns units.
    pub start_time: u64,
    /// End of the segment, in 100ns units.
    pub end_time: u64,
    /// Grain parameters applied to the whole segment.
    pub grain_params: FilmGrainParams,
}

impl From<av1_grain::GrainTableSegment> for GrainTableSegment {
    fn from(data: av1_grain::GrainTableSegment) -> Self {
        GrainTableSegment {
            start_time: data.start_time,
            end_time: data.end_time,
            grain_params: data.into(),
        }
    }
}

/// Options for [`analyze`].
#[derive(Debug, Clone)]
pub struct AnalyzeOptions {
    /// The untouched source file.
    pub source: PathBuf,
    /// The denoised version of the source file.
    pub denoised: PathBuf,
    /// The path to write the output film grain table to.
    pub output: PathBuf,
    /// Overwrite the output file if it exists. When `false` and the output
    /// exists, [`analyze`] returns an error instead of prompting.
    pub overwrite: bool,
    /// A semicolon-separated list of filters to apply to the source before
    /// running the diff (e.g. `"crop:top=42,left=64;resize:width=1920,height=1080"`).
    pub filters: Option<String>,
    /// Number of Rayon worker threads for grain analysis. `None` keeps the
    /// default global pool. A pool that is already initialized is left as-is.
    pub threads: Option<usize>,
}

/// Outcome of [`analyze`].
#[derive(Debug, Clone, Copy)]
pub struct AnalyzeReport {
    /// Number of frame pairs the grain table was computed from.
    pub frames: usize,
}

/// Generate a film grain table by diffing a source video against a denoised copy.
///
/// This produces the most accurate grain table because it measures the actual
/// noise present in the source rather than estimating it.
///
/// Unlike the CLI, this never prompts: an existing output without
/// `overwrite` is an error, and identical input paths are an error.
pub fn analyze(opts: &AnalyzeOptions) -> Result<AnalyzeReport> {
    if opts.source == opts.output || opts.denoised == opts.output {
        bail!("Input and output paths are the same, which would overwrite an input");
    }

    if opts.source == opts.denoised {
        bail!("Source and denoised paths are the same, which would compute an empty diff");
    }

    let filters = match &opts.filters {
        Some(f) => Some(FilterChain::new(f).map_err(|e| anyhow!("Invalid filter chain: {e}"))?),
        None => None,
    };

    if let Some(threads) = opts.threads {
        if threads == 0 {
            bail!("Thread count must be greater than zero");
        }
        // A process-wide Rayon pool can only be initialized once; a pool that
        // already exists (e.g. from an earlier call in the same process) wins.
        let _ = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global();
    }

    if opts.output.exists() && !opts.overwrite {
        bail!(
            "Output file {} exists and overwriting was not requested",
            opts.output.to_string_lossy()
        );
    }

    let frame_count = get_frame_count(&opts.source).ok();

    let progress = if stderr().is_tty() {
        let pb = frame_count.map_or_else(
            || ProgressBar::new(0).with_style(spinner_style()),
            |frame_count| {
                ProgressBar::new(frame_count as u64).with_style(pretty_progress_style())
            },
        );
        pb.set_draw_target(ProgressDrawTarget::stderr());
        pb.enable_steady_tick(Duration::from_millis(100));
        pb.reset();
        pb.reset_eta();
        pb.reset_elapsed();
        pb.set_position(0);
        pb
    } else {
        ProgressBar::hidden()
    };

    let source_reader = BitstreamReader::open(&opts.source)?;
    let denoised_reader = BitstreamReader::open(&opts.denoised)?;
    let frame_rate = source_reader.get_video_details().frame_rate;
    let source_bd = source_reader.get_video_details().bit_depth;
    let denoised_bd = denoised_reader.get_video_details().bit_depth;
    let can_preconvert_to_u8 = filters.as_ref().is_none_or(FilterChain::is_crop_only);
    let differ_source_bd = if can_preconvert_to_u8 { 8 } else { source_bd };
    let differ_denoised_bd = if can_preconvert_to_u8 { 8 } else { denoised_bd };
    let mut differ = DiffGenerator::new(
        num_rational::Rational64::new(
            i64::from(*frame_rate.numer()),
            i64::from(*frame_rate.denom()),
        ),
        differ_source_bd,
        differ_denoised_bd,
    );
    let non_zero_source_bd =
        NonZeroU8::new(source_bd as u8).ok_or_else(|| anyhow!("bd should not be 0"))?;

    let frames = profile::time(
        profile::MetricId::DiffFramePairsTotal,
        || -> Result<usize> {
            if can_preconvert_to_u8 {
                return diff_frame_pairs_preconverted_u8(
                    source_reader,
                    denoised_reader,
                    filters,
                    &mut differ,
                    &progress,
                );
            }

            match (source_bd, denoised_bd) {
                (8, 8) => diff_frame_pairs::<u8, u8>(
                    source_reader,
                    denoised_reader,
                    non_zero_source_bd,
                    filters,
                    &mut differ,
                    &progress,
                ),
                (8, 9..=16) => diff_frame_pairs::<u8, u16>(
                    source_reader,
                    denoised_reader,
                    non_zero_source_bd,
                    filters,
                    &mut differ,
                    &progress,
                ),
                (9..=16, 8) => diff_frame_pairs::<u16, u8>(
                    source_reader,
                    denoised_reader,
                    non_zero_source_bd,
                    filters,
                    &mut differ,
                    &progress,
                ),
                (9..=16, 9..=16) => diff_frame_pairs::<u16, u16>(
                    source_reader,
                    denoised_reader,
                    non_zero_source_bd,
                    filters,
                    &mut differ,
                    &progress,
                ),
                _ => {
                    bail!("Bit depths not between 8-16 are not currently supported");
                }
            }
        },
    )?;
    progress.finish();

    let grain_tables = profile::time(profile::MetricId::DifferFinish, || differ.finish());
    profile::time(profile::MetricId::OutputWrite, || -> Result<()> {
        let mut output_file = BufWriter::new(File::create(&opts.output)?);
        writeln!(&mut output_file, "filmgrn1")?;
        for segment in grain_tables {
            write_film_grain_segment(&segment.into(), &mut output_file)?;
        }
        output_file.flush()?;
        Ok(())
    })?;
    profile::print_report();
    info!("Computed diff for {frames} frames");
    info!("Done, wrote output file to {}", opts.output.to_string_lossy());
    Ok(AnalyzeReport { frames })
}

/// Where the grain applied by [`apply`] comes from.
#[derive(Debug, Clone)]
pub enum GrainSource {
    /// Path to a `filmgrn1` grain table file.
    TableFile(PathBuf),
    /// Name of a built-in film stock preset (case-insensitive).
    Preset(String),
    /// Photon-noise-based grain generated for the input video.
    PhotonNoise(PhotonNoiseOptions),
}

/// Options for photon-noise grain generation in [`apply`].
#[derive(Debug, Clone, Copy)]
pub struct PhotonNoiseOptions {
    /// ISO strength (1–4294967295; 100–6400 recommended).
    pub iso: u32,
    /// Apply grain to chroma planes as well as luma.
    pub chroma: bool,
}

/// Options for [`apply`].
#[derive(Debug, Clone)]
pub struct ApplyOptions {
    /// The AV1 or HEVC file to apply grain to.
    pub input: PathBuf,
    /// The path to write the grain-synthed file to.
    pub output: PathBuf,
    /// Overwrite the output file if it exists. When `false` and the output
    /// exists, [`apply`] returns an error instead of prompting.
    pub overwrite: bool,
    /// Where the grain comes from.
    pub source: GrainSource,
    /// Overwrite grain metadata already present in the input. Without this,
    /// inputs that already carry grain are skipped and reported.
    pub replace: bool,
}

/// Outcome of [`apply`].
#[derive(Debug, Clone, Copy)]
pub struct ApplyReport {
    /// `true` when the input already carried grain and `replace` was `false`,
    /// so nothing was written.
    pub skipped_existing: bool,
}

/// Apply film grain to an AV1 or HEVC video and write the result to `output`.
///
/// The bitstream is edited in place — no re-encoding happens. For AV1 inputs
/// this edits film grain headers; for HEVC inputs it injects x265-compatible
/// AFGS1 prefix SEI messages.
///
/// Unlike the CLI, this never prompts: an existing output without
/// `overwrite` is an error, and identical input/output paths are an error.
pub fn apply(opts: &ApplyOptions) -> Result<ApplyReport> {
    if opts.input == opts.output {
        bail!("Input and output paths are the same, which would overwrite the input");
    }

    if opts.output.exists() && !opts.overwrite {
        bail!(
            "Output file {} exists and overwriting was not requested",
            opts.output.to_string_lossy()
        );
    }

    let reader = BitstreamReader::open(&opts.input)?;
    let codec_id = reader.get_video_stream()?.parameters().id();

    if codec_id != codec::Id::HEVC {
        // Check whether the input already carries AV1 film grain headers.
        // We only need to read the Sequence Header OBU (always in the first video packet)
        // to check the film_grain_params_present flag, so this is effectively instant.
        let check_reader = BitstreamReader::open(&opts.input)?;
        let mut check_parser: BitstreamParser<false> = BitstreamParser::new(check_reader);
        let has_existing_grain = check_parser.film_grain_params_present()?;

        if has_existing_grain && !opts.replace {
            info!(
                "Skipped: grain headers already exist in this file. Re-run with replace \
                 enabled to replace the existing grain headers."
            );
            return Ok(ApplyReport {
                skipped_existing: true,
            });
        }
    }

    // Build the grain segments from whichever source was provided.
    let new_grain = match &opts.source {
        GrainSource::TableFile(grain_path) => {
            let grain_data = read_to_string(grain_path)?;
            let new_headers = parse_grain_table(&grain_data)?;
            Some(
                new_headers
                    .into_iter()
                    .map(|h| h.into())
                    .collect::<Vec<_>>(),
            )
        }
        GrainSource::Preset(preset_name) => {
            let grain_data = presets::get_preset(preset_name).ok_or_else(|| {
                // Enumerate valid names: standalones + format bases + format+suffix combos.
                let mut known: Vec<String> = presets::STANDALONE_PRESETS
                    .iter()
                    .map(|p| p.name.to_string())
                    .collect();
                for fp in presets::FORMAT_PRESETS {
                    known.push(fp.name.to_string());
                    for stock in &presets::FILM_STOCKS[1..] {
                        known.push(format!("{}{}", fp.name, stock.suffix));
                    }
                }
                anyhow!(
                    "Unknown preset '{preset_name}'. Valid presets are:\n  {}",
                    known.join(", ")
                )
            })?;
            let new_headers = parse_grain_table(grain_data)?;
            Some(
                new_headers
                    .into_iter()
                    .map(|h| h.into())
                    .collect::<Vec<_>>(),
            )
        }
        GrainSource::PhotonNoise(noise) => {
            // SAFETY: We extract the items we need from the struct within the unsafe
            // block, so there's no possibility of use-after-free later.
            let (width, height, trc, range) = unsafe {
                let video_stream = reader.get_video_stream().unwrap();
                let params = video_stream.parameters().as_ptr();
                (
                    (*params).width as u32,
                    (*params).height as u32,
                    (*params).color_trc,
                    (*params).color_range,
                )
            };
            let grain_data = generate_photon_noise_params(
                0,
                u64::MAX,
                av1_grain::NoiseGenArgs {
                    iso_setting: noise.iso,
                    width,
                    height,
                    transfer_function: if trc == AVColorTransferCharacteristic::SMPTE2084 {
                        TransferFunction::SMPTE2084
                    } else {
                        TransferFunction::BT1886
                    },
                    chroma_grain: noise.chroma,
                    full_range: range == AVColorRange::JPEG,
                    random_seed: None,
                },
            );
            Some(vec![grain_data.into()])
        }
    };

    if codec_id == codec::Id::HEVC {
        if !opts.replace && hevc::has_hevc_grain(&opts.input)? {
            info!(
                "Skipped: HEVC stream already contains x265 AFGS1 SEI. Re-run with replace \
                 enabled to replace the existing grain SEI."
            );
            return Ok(ApplyReport {
                skipped_existing: true,
            });
        }
        hevc::modify_hevc_grain(&opts.input, &opts.output, new_grain.as_deref(), opts.replace)?;
        info!("Done, wrote output file to {}", opts.output.to_string_lossy());
        return Ok(ApplyReport {
            skipped_existing: false,
        });
    }

    let writer = format::output(&opts.output)?;

    let mut parser: BitstreamParser<true> =
        BitstreamParser::with_writer(reader, writer, new_grain);

    parser.modify_grain_headers()?;

    info!("Done, wrote output file to {}", opts.output.to_string_lossy());
    Ok(ApplyReport {
        skipped_existing: false,
    })
}

const PROGRESS_CHARS: &str = "█▉▊▋▌▍▎▏  ";
const INDICATIF_PROGRESS_TEMPLATE: &str = if cfg!(windows) {
    // Do not use a spinner on Windows since the default console cannot display
    // the characters used for the spinner
    "{elapsed_precise:.bold} ▕{wide_bar:.blue/white.dim}▏ {percent:.bold}  {pos} ({fps:.bold}, eta {fixed_eta}{msg})"
} else {
    "{spinner:.green.bold} {elapsed_precise:.bold} ▕{wide_bar:.blue/white.dim}▏ {percent:.bold}  {pos} ({fps:.bold}, eta {fixed_eta}{msg})"
};
const INDICATIF_SPINNER_TEMPLATE: &str = if cfg!(windows) {
    // Do not use a spinner on Windows since the default console cannot display
    // the characters used for the spinner
    "{elapsed_precise:.bold} [{wide_bar:.blue/white.dim}]  {pos} frames ({fps:.bold})"
} else {
    "{spinner:.green.bold} [{wide_bar:.blue/white.dim}]  {pos} frames ({fps:.bold})"
};

fn pretty_progress_style() -> ProgressStyle {
    ProgressStyle::default_bar()
        .template(INDICATIF_PROGRESS_TEMPLATE)
        .unwrap()
        .with_key(
            "fps",
            |state: &ProgressState, w: &mut dyn std::fmt::Write| {
                if state.pos() == 0 || state.elapsed().as_secs_f32() < f32::EPSILON {
                    write!(w, "0 fps").unwrap();
                } else {
                    let fps = state.pos() as f32 / state.elapsed().as_secs_f32();
                    if fps < 1.0 {
                        write!(w, "{:.2} s/fr", 1.0 / fps).unwrap();
                    } else {
                        write!(w, "{fps:.2} fps").unwrap();
                    }
                }
            },
        )
        .with_key(
            "fixed_eta",
            |state: &ProgressState, w: &mut dyn std::fmt::Write| {
                if state.pos() == 0 || state.elapsed().as_secs_f32() < f32::EPSILON {
                    write!(w, "unknown").unwrap();
                } else {
                    let spf = state.elapsed().as_secs_f32() / state.pos() as f32;
                    let remaining = state.len().unwrap_or(0) - state.pos();
                    write!(
                        w,
                        "{:#}",
                        HumanDuration(Duration::from_secs_f32(spf * remaining as f32))
                    )
                    .unwrap();
                }
            },
        )
        .with_key(
            "pos",
            |state: &ProgressState, w: &mut dyn std::fmt::Write| {
                write!(w, "{}/{}", state.pos(), state.len().unwrap_or(0)).unwrap();
            },
        )
        .with_key(
            "percent",
            |state: &ProgressState, w: &mut dyn std::fmt::Write| {
                write!(w, "{:>3.0}%", state.fraction() * 100_f32).unwrap();
            },
        )
        .progress_chars(PROGRESS_CHARS)
}

fn spinner_style() -> ProgressStyle {
    ProgressStyle::default_spinner()
        .template(INDICATIF_SPINNER_TEMPLATE)
        .unwrap()
        .with_key(
            "fps",
            |state: &ProgressState, w: &mut dyn std::fmt::Write| {
                if state.pos() == 0 || state.elapsed().as_secs_f32() < f32::EPSILON {
                    write!(w, "0 fps").unwrap();
                } else {
                    let fps = state.pos() as f32 / state.elapsed().as_secs_f32();
                    if fps < 1.0 {
                        write!(w, "{:.2} s/fr", 1.0 / fps).unwrap();
                    } else {
                        write!(w, "{fps:.2} fps").unwrap();
                    }
                }
            },
        )
        .with_key(
            "pos",
            |state: &ProgressState, w: &mut dyn std::fmt::Write| {
                write!(w, "{}", state.pos()).unwrap();
            },
        )
        .progress_chars(PROGRESS_CHARS)
}

#[allow(clippy::type_complexity)]
fn diff_frame_pairs<T: Pixel + Pixel05 + Send + 'static, U: Pixel + Pixel05 + Send + 'static>(
    mut source_reader: BitstreamReader,
    mut denoised_reader: BitstreamReader,
    source_bd: NonZeroU8,
    filters: Option<FilterChain>,
    differ: &mut DiffGenerator,
    progress: &ProgressBar,
) -> Result<usize> {
    let (source_sender, source_receiver) = sync_channel::<Result<Option<Frame<T>>>>(1);
    let source_producer = std::thread::spawn(move || {
        loop {
            let mut frame = profile::time(profile::MetricId::SourceGetFrame, || {
                source_reader.get_frame::<T>()
            });
            if let Some(f) = filters.as_ref() {
                frame = profile::time(profile::MetricId::SourceFilter, || {
                    frame.map(|opt| opt.map(|source_frame| f.apply(source_frame, source_bd)))
                });
            }
            let should_stop = !matches!(frame, Ok(Some(_)));

            if source_sender.send(frame).is_err() || should_stop {
                break;
            }
        }
    });
    let (denoised_sender, denoised_receiver) = sync_channel::<Result<Option<Frame<U>>>>(1);
    let denoised_producer = std::thread::spawn(move || {
        loop {
            let frame = profile::time(profile::MetricId::DenoisedGetFrame, || {
                denoised_reader.get_frame::<U>()
            });
            let should_stop = !matches!(frame, Ok(Some(_)));

            if denoised_sender.send(frame).is_err() || should_stop {
                break;
            }
        }
    });

    let mut frames = 0usize;
    let mut result = Ok(());
    loop {
        debug!("Diffing next frame");
        let source_msg = profile::time(profile::MetricId::ConsumerRecv, || source_receiver.recv());
        let denoised_msg =
            profile::time(profile::MetricId::ConsumerRecv, || denoised_receiver.recv());
        match (source_msg, denoised_msg) {
            (Ok(Ok(Some(source_frame))), Ok(Ok(Some(denoised_frame)))) => {
                if let Err(e) = profile::time(profile::MetricId::DifferDiffFrame, || {
                    differ.diff_frame(&source_frame, &denoised_frame)
                }) {
                    result = Err(e);
                    break;
                }
                frames += 1;
                progress.inc(1);
            }
            (Ok(Ok(None)), Ok(Ok(None))) => {
                break;
            }
            (Ok(Ok(_)), Ok(Ok(_))) => {
                warn!(
                    "Videos did not have equal frame counts. Resulting grain table may not be as \
                     expected."
                );
                break;
            }
            (Ok(Err(e)), _) | (_, Ok(Err(e))) => {
                result = Err(e);
                break;
            }
            (Err(_), _) | (_, Err(_)) => {
                break;
            }
        }
    }

    drop(source_receiver);
    drop(denoised_receiver);
    source_producer
        .join()
        .map_err(|_| anyhow!("source read-ahead thread panicked"))?;
    denoised_producer
        .join()
        .map_err(|_| anyhow!("denoised read-ahead thread panicked"))?;
    result?;

    Ok(frames)
}

fn diff_frame_pairs_preconverted_u8(
    mut source_reader: BitstreamReader,
    mut denoised_reader: BitstreamReader,
    filters: Option<FilterChain>,
    differ: &mut DiffGenerator,
    progress: &ProgressBar,
) -> Result<usize> {
    let (source_sender, source_receiver) = sync_channel::<Result<Option<Frame<u8>>>>(1);
    let source_producer = std::thread::spawn(move || {
        let filter_bd = NonZeroU8::new(8).expect("non-zero constant");
        loop {
            let mut frame = profile::time(profile::MetricId::SourceGetFrame, || {
                source_reader.get_frame_u8()
            });
            if let Some(f) = filters.as_ref() {
                frame = profile::time(profile::MetricId::SourceFilter, || {
                    frame.map(|opt| opt.map(|source_frame| f.apply(source_frame, filter_bd)))
                });
            }
            let should_stop = !matches!(frame, Ok(Some(_)));

            if source_sender.send(frame).is_err() || should_stop {
                break;
            }
        }
    });
    let (denoised_sender, denoised_receiver) = sync_channel::<Result<Option<Frame<u8>>>>(1);
    let denoised_producer = std::thread::spawn(move || {
        loop {
            let frame = profile::time(profile::MetricId::DenoisedGetFrame, || {
                denoised_reader.get_frame_u8()
            });
            let should_stop = !matches!(frame, Ok(Some(_)));

            if denoised_sender.send(frame).is_err() || should_stop {
                break;
            }
        }
    });

    let mut frames = 0usize;
    let mut result = Ok(());
    loop {
        debug!("Diffing next frame");
        let source_msg = profile::time(profile::MetricId::ConsumerRecv, || source_receiver.recv());
        let denoised_msg =
            profile::time(profile::MetricId::ConsumerRecv, || denoised_receiver.recv());
        match (source_msg, denoised_msg) {
            (Ok(Ok(Some(source_frame))), Ok(Ok(Some(denoised_frame)))) => {
                if let Err(e) = profile::time(profile::MetricId::DifferDiffFrame, || {
                    differ.diff_frame(&source_frame, &denoised_frame)
                }) {
                    result = Err(e);
                    break;
                }
                frames += 1;
                progress.inc(1);
            }
            (Ok(Ok(None)), Ok(Ok(None))) => {
                break;
            }
            (Ok(Ok(_)), Ok(Ok(_))) => {
                warn!(
                    "Videos did not have equal frame counts. Resulting grain table may not be as \
                     expected."
                );
                break;
            }
            (Ok(Err(e)), _) | (_, Ok(Err(e))) => {
                result = Err(e);
                break;
            }
            (Err(_), _) | (_, Err(_)) => {
                break;
            }
        }
    }

    drop(source_receiver);
    drop(denoised_receiver);
    source_producer
        .join()
        .map_err(|_| anyhow!("source read-ahead thread panicked"))?;
    denoised_producer
        .join()
        .map_err(|_| anyhow!("denoised read-ahead thread panicked"))?;
    result?;

    Ok(frames)
}

/// Write one timestamped grain segment in `filmgrn1` table text format.
///
/// Also used by the binary's `inspect` command.
pub fn write_film_grain_segment(
    segment: &GrainTableSegment,
    output: &mut BufWriter<File>,
) -> anyhow::Result<()> {
    let params = &segment.grain_params;

    writeln!(
        output,
        "E {} {} 1 {} 1",
        segment.start_time, segment.end_time, params.grain_seed,
    )?;
    writeln!(
        output,
        "\tp {} {} {} {} {} {} {} {} {} {} {} {}",
        params.ar_coeff_lag,
        params.ar_coeff_shift,
        params.grain_scale_shift,
        params.scaling_shift,
        u8::from(params.chroma_scaling_from_luma),
        u8::from(params.overlap_flag),
        params.cb_mult,
        params.cb_luma_mult,
        params.cb_offset,
        params.cr_mult,
        params.cr_luma_mult,
        params.cr_offset
    )?;

    write!(output, "\tsY {} ", params.scaling_points_y.len())?;
    for point in &params.scaling_points_y {
        write!(output, " {} {}", point[0], point[1])?;
    }
    writeln!(output)?;

    write!(output, "\tsCb {}", params.scaling_points_cb.len())?;
    for point in &params.scaling_points_cb {
        write!(output, " {} {}", point[0], point[1])?;
    }
    writeln!(output)?;

    write!(output, "\tsCr {}", params.scaling_points_cr.len())?;
    for point in &params.scaling_points_cr {
        write!(output, " {} {}", point[0], point[1])?;
    }
    writeln!(output)?;

    write!(output, "\tcY")?;
    for coeff in &params.ar_coeffs_y {
        write!(output, " {}", *coeff)?;
    }
    writeln!(output)?;

    write!(output, "\tcCb")?;
    for coeff in &params.ar_coeffs_cb {
        write!(output, " {}", *coeff)?;
    }
    writeln!(output)?;

    write!(output, "\tcCr")?;
    for coeff in &params.ar_coeffs_cr {
        write!(output, " {}", *coeff)?;
    }
    writeln!(output)?;

    Ok(())
}

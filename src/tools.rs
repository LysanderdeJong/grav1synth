//! Everything grav1synth needs from the `ffmpeg`/`ffprobe` executables.
//!
//! The library never starts processes itself: every launch goes through a [`Launcher`], so an
//! embedding application can route it through its own process management (configured paths,
//! cancellation, fake runners in tests). [`StdLauncher`] is the plain `std::process` launcher
//! used by the CLI.

use std::{
    ffi::OsString,
    io::{BufReader, Read, Write},
    path::Path,
    process::{Command, Stdio},
    thread::JoinHandle,
};

use anyhow::{Context, Result, anyhow, bail};
use av1_grain::v_frame::chroma::ChromaSubsampling;
use num_rational::Rational32;

/// Which executable to start.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tool {
    Ffmpeg,
    Ffprobe,
}

/// Starts `ffmpeg`/`ffprobe`. stdout is always piped; stdin is piped when asked for.
pub trait Launcher: Sync {
    /// Start `tool` with `args`.
    ///
    /// # Errors
    /// Returns an error when the process cannot be started.
    fn spawn(&self, tool: Tool, args: &[OsString], stdin: bool) -> Result<Box<dyn Process>>;
}

/// A started process.
pub trait Process: Send {
    /// Take the stdin pipe (only present when requested at spawn).
    fn stdin(&mut self) -> Option<Box<dyn Write + Send>>;
    /// Take the stdout pipe.
    fn stdout(&mut self) -> Option<Box<dyn Read + Send>>;
    /// Wait for exit.
    ///
    /// # Errors
    /// Returns an error naming the cause (stderr) when the process failed.
    fn wait(self: Box<Self>) -> Result<()>;
}

/// Launches `ffmpeg`/`ffprobe` from `PATH` with `std::process`.
pub struct StdLauncher;

impl Launcher for StdLauncher {
    fn spawn(&self, tool: Tool, args: &[OsString], stdin: bool) -> Result<Box<dyn Process>> {
        let name = match tool {
            Tool::Ffmpeg => "ffmpeg",
            Tool::Ffprobe => "ffprobe",
        };
        let mut child = Command::new(name)
            .args(args)
            .stdin(if stdin { Stdio::piped() } else { Stdio::null() })
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .with_context(|| format!("start {name} (is it on PATH?)"))?;
        let mut stderr = child.stderr.take().expect("stderr is piped");
        let stderr = std::thread::spawn(move || {
            let mut text = String::new();
            let _ = stderr.read_to_string(&mut text);
            text
        });
        Ok(Box::new(StdProcess {
            name,
            child,
            stderr,
        }))
    }
}

struct StdProcess {
    name: &'static str,
    child: std::process::Child,
    stderr: JoinHandle<String>,
}

impl Process for StdProcess {
    fn stdin(&mut self) -> Option<Box<dyn Write + Send>> {
        self.child
            .stdin
            .take()
            .map(|s| Box::new(s) as Box<dyn Write + Send>)
    }

    fn stdout(&mut self) -> Option<Box<dyn Read + Send>> {
        self.child
            .stdout
            .take()
            .map(|s| Box::new(s) as Box<dyn Read + Send>)
    }

    fn wait(mut self: Box<Self>) -> Result<()> {
        let status = self.child.wait()?;
        let stderr = self.stderr.join().unwrap_or_default();
        if !status.success() {
            bail!("{} failed ({status}): {}", self.name, stderr.trim());
        }
        Ok(())
    }
}

/// Run `tool` to completion and return its stdout.
fn run(launcher: &dyn Launcher, tool: Tool, args: &[OsString]) -> Result<String> {
    let mut process = launcher.spawn(tool, args, false)?;
    let mut out = String::new();
    let read = process
        .stdout()
        .ok_or_else(|| anyhow!("no stdout pipe"))?
        .read_to_string(&mut out);
    process.wait()?;
    read?;
    Ok(out)
}

fn os(args: &[&str], input: &Path) -> Vec<OsString> {
    let mut out: Vec<OsString> = args.iter().map(OsString::from).collect();
    out.push(input.into());
    out
}

/// What grav1synth needs to know about a file and its first (non-cover-art) video stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MediaInfo {
    /// Index of the video stream in the file.
    pub video_index: usize,
    /// `ffprobe` codec name of the video stream (`av1`, `hevc`, ...).
    pub codec: String,
    pub width: usize,
    pub height: usize,
    pub pix_fmt: String,
    /// `avg_frame_rate`, or `r_frame_rate` when the average is unknown.
    pub frame_rate: Rational32,
    /// `r_frame_rate` as (num, den); (0, 0) when unknown.
    pub r_frame_rate: (u64, u64),
    /// Video stream time base as (num, den).
    pub time_base: (u64, u64),
    pub color_transfer: String,
    pub color_range: String,
    pub has_b_frames: u64,
    /// Sample aspect ratio as (num, den); (0, 0) when unknown.
    pub sample_aspect_ratio: (u64, u64),
    pub extradata: Vec<u8>,
    /// `DISPOSITION:*` flags set on the video stream.
    pub disposition: Vec<String>,
    /// Every stream of the file: (index, codec_type).
    pub streams: Vec<(usize, String)>,
}

impl MediaInfo {
    /// Bit depth and chroma subsampling of the video's pixel format.
    ///
    /// # Errors
    /// Returns an error for pixel formats grav1synth cannot read.
    pub fn pixel_layout(&self) -> Result<(usize, ChromaSubsampling)> {
        Ok(match self.pix_fmt.as_str() {
            "yuv420p" | "yuvj420p" => (8, ChromaSubsampling::Yuv420),
            "yuv422p" | "yuvj422p" => (8, ChromaSubsampling::Yuv422),
            "yuv444p" | "yuvj444p" => (8, ChromaSubsampling::Yuv444),
            "yuv420p10le" => (10, ChromaSubsampling::Yuv420),
            "yuv422p10le" => (10, ChromaSubsampling::Yuv422),
            "yuv444p10le" => (10, ChromaSubsampling::Yuv444),
            "yuv420p12le" => (12, ChromaSubsampling::Yuv420),
            "yuv422p12le" => (12, ChromaSubsampling::Yuv422),
            "yuv444p12le" => (12, ChromaSubsampling::Yuv444),
            fmt => bail!("unsupported video format {fmt}"),
        })
    }
}

/// Probe `input` with `ffprobe`.
///
/// # Errors
/// Returns an error when ffprobe fails or the file has no video stream.
pub fn probe(launcher: &dyn Launcher, input: &Path) -> Result<MediaInfo> {
    let streams = run(
        launcher,
        Tool::Ffprobe,
        &os(
            &[
                "-v",
                "error",
                "-show_entries",
                "stream=index,codec_type",
                "-of",
                "csv=p=0",
            ],
            input,
        ),
    )?;
    let streams = streams
        .lines()
        .filter_map(|l| {
            let (i, t) = l.trim().split_once(',')?;
            Some((i.parse().ok()?, t.to_owned()))
        })
        .collect();
    let text = run(
        launcher,
        Tool::Ffprobe,
        &os(
            &[
                "-v",
                "error",
                "-select_streams",
                "V:0",
                "-show_entries",
                "stream=index,codec_name,width,height,pix_fmt,avg_frame_rate,r_frame_rate,\
                 time_base,color_transfer,color_range,has_b_frames,sample_aspect_ratio,\
                 extradata:stream_disposition",
                "-show_data",
                "-of",
                "default=nw=1",
            ],
            input,
        ),
    )?;
    parse_probe(&text, streams).with_context(|| format!("probe {}", input.display()))
}

fn parse_probe(text: &str, streams: Vec<(usize, String)>) -> Result<MediaInfo> {
    let mut lines = text.lines().map(str::trim_end).peekable();
    let mut field = std::collections::HashMap::new();
    let mut extradata = Vec::new();
    let mut disposition = Vec::new();
    while let Some(line) = lines.next() {
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        if key == "extradata" {
            // ffprobe hex dump: "00000000: 0101 6000 ...  ascii"
            while let Some(hex) = lines.peek().and_then(|l| l.get(10..49)) {
                if !lines
                    .peek()
                    .is_some_and(|l| l.as_bytes().get(8) == Some(&b':'))
                {
                    break;
                }
                let hex: String = hex.chars().filter(|c| !c.is_whitespace()).collect();
                for pair in hex.as_bytes().chunks(2) {
                    extradata.push(u8::from_str_radix(std::str::from_utf8(pair)?, 16)?);
                }
                lines.next();
            }
        } else if let Some(flag) = key.strip_prefix("DISPOSITION:") {
            if value == "1" {
                disposition.push(flag.to_owned());
            }
        } else {
            field.insert(key.to_owned(), value.to_owned());
        }
    }
    let get = |k: &str| {
        field
            .get(k)
            .cloned()
            .ok_or_else(|| anyhow!("no video stream found"))
    };
    let num = |k: &str| -> Result<u64> { Ok(get(k)?.parse().unwrap_or(0)) };
    let ratio = |k: &str, sep: char| -> (u64, u64) {
        get(k)
            .ok()
            .and_then(|v| {
                let (n, d) = v.split_once(sep)?;
                Some((n.parse().ok()?, d.parse().ok()?))
            })
            .unwrap_or((0, 0))
    };
    let mut frame_rate = ratio("avg_frame_rate", '/');
    if frame_rate.1 == 0 {
        frame_rate = ratio("r_frame_rate", '/');
    }
    Ok(MediaInfo {
        video_index: num("index")? as usize,
        codec: get("codec_name")?,
        width: num("width")? as usize,
        height: num("height")? as usize,
        pix_fmt: get("pix_fmt")?,
        frame_rate: Rational32::new_raw(frame_rate.0 as i32, frame_rate.1 as i32),
        r_frame_rate: ratio("r_frame_rate", '/'),
        time_base: ratio("time_base", '/'),
        color_transfer: get("color_transfer")?,
        color_range: get("color_range")?,
        has_b_frames: num("has_b_frames")?,
        sample_aspect_ratio: ratio("sample_aspect_ratio", ':'),
        extradata,
        disposition,
        streams,
    })
}

/// Number of packets in the first video stream (`ffprobe -count_packets`).
///
/// # Errors
/// Returns an error when ffprobe fails or prints no count.
pub fn frame_count(launcher: &dyn Launcher, video: &Path) -> Result<usize> {
    let out = run(
        launcher,
        Tool::Ffprobe,
        &os(
            &[
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_packets",
                "-show_entries",
                "stream=nb_read_packets",
                "-of",
                "csv=p=0",
            ],
            video,
        ),
    )?;
    Ok(out.trim().parse()?)
}

/// One demuxed packet of the video stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Packet {
    pub data: Vec<u8>,
    pub pts: Option<i64>,
    pub dts: Option<i64>,
    pub duration: i64,
    pub key: bool,
}

/// Timing of one packet as listed by ffprobe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PacketMeta {
    pts: Option<i64>,
    dts: Option<i64>,
    duration: i64,
    size: usize,
    key: bool,
}

/// Reads the video stream's packets: timing from `ffprobe -show_packets`, payloads from
/// `ffmpeg -c copy -f data`, matched one-to-one by order.
pub struct PacketReader {
    metas: Vec<PacketMeta>,
    next: usize,
    payload: BufReader<Box<dyn Read + Send>>,
    process: Option<Box<dyn Process>>,
}

impl PacketReader {
    /// Start reading the packets of `info.video_index` in `input`.
    ///
    /// # Errors
    /// Returns an error when ffprobe or ffmpeg cannot be started or ffprobe fails.
    pub fn open(launcher: &dyn Launcher, input: &Path, info: &MediaInfo) -> Result<Self> {
        let listing = run(
            launcher,
            Tool::Ffprobe,
            &os(
                &[
                    "-v",
                    "error",
                    "-select_streams",
                    &info.video_index.to_string(),
                    "-show_entries",
                    "packet=pts,dts,duration,size,flags",
                    "-of",
                    "compact=p=0",
                ],
                input,
            ),
        )?;
        let metas = parse_packet_listing(&listing)?;
        let mut args = os(&["-nostdin", "-v", "error", "-copyts", "-i"], input);
        for arg in [
            "-map",
            &format!("0:{}", info.video_index),
            "-c",
            "copy",
            "-copyinkf",
            "-f",
            "data",
            "-",
        ] {
            args.push(arg.into());
        }
        let mut process = launcher.spawn(Tool::Ffmpeg, &args, false)?;
        let payload = process.stdout().ok_or_else(|| anyhow!("no stdout pipe"))?;
        Ok(Self {
            metas,
            next: 0,
            payload: BufReader::with_capacity(1 << 20, payload),
            process: Some(process),
        })
    }

    /// The next packet, or `None` after the last one.
    ///
    /// # Errors
    /// Returns an error when ffmpeg fails or delivers a different amount of data than ffprobe
    /// listed.
    pub fn next_packet(&mut self) -> Result<Option<Packet>> {
        let Some(&meta) = self.metas.get(self.next) else {
            let mut rest = [0u8; 1];
            let extra = self.payload.read(&mut rest)?;
            if let Some(process) = self.process.take() {
                process.wait()?;
            }
            if extra != 0 {
                bail!("ffmpeg delivered more video data than ffprobe listed");
            }
            return Ok(None);
        };
        self.next += 1;
        let mut data = vec![0; meta.size];
        if let Err(e) = self.payload.read_exact(&mut data) {
            if let Some(process) = self.process.take() {
                process.wait()?;
            }
            return Err(anyhow!(e).context("video stream ended early (truncated input?)"));
        }
        Ok(Some(Packet {
            data,
            pts: meta.pts,
            dts: meta.dts,
            duration: meta.duration,
            key: meta.key,
        }))
    }

    /// Number of packets ffprobe listed.
    #[must_use]
    pub fn len(&self) -> usize {
        self.metas.len()
    }

    /// Whether the stream has no packets.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.metas.is_empty()
    }

    /// What [`Remuxer`] restores that NUT does not carry: the last packet's duration and the
    /// dts of the first `decode_delay` packets (later ones are rebuilt from the pts order).
    #[must_use]
    pub fn timing(&self, decode_delay: u64) -> RemuxTiming {
        RemuxTiming {
            packets: self.metas.len(),
            last_duration: self.metas.last().map_or(0, |m| m.duration),
            leading_dts: self
                .metas
                .iter()
                .take(decode_delay as usize)
                .map_while(|m| m.dts)
                .collect(),
        }
    }

    /// Stop early. ffmpeg's resulting broken-pipe exit is not an error.
    pub fn stop(mut self) {
        if let Some(process) = self.process.take() {
            drop(std::mem::replace(
                &mut self.payload,
                BufReader::new(Box::new(std::io::empty())),
            ));
            let _ = process.wait();
        }
    }
}

impl Iterator for PacketReader {
    type Item = Result<Packet>;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_packet().transpose()
    }
}

fn parse_packet_listing(text: &str) -> Result<Vec<PacketMeta>> {
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|line| {
            let mut meta = PacketMeta {
                pts: None,
                dts: None,
                duration: 0,
                size: 0,
                key: false,
            };
            for kv in line.trim().split('|') {
                let (k, v) = kv.split_once('=').unwrap_or((kv, ""));
                match k {
                    "pts" => meta.pts = v.parse().ok(),
                    "dts" => meta.dts = v.parse().ok(),
                    "duration" => meta.duration = v.parse().unwrap_or(0),
                    "size" => {
                        meta.size = v
                            .parse()
                            .with_context(|| format!("bad packet line {line:?}"))?;
                    }
                    "flags" => meta.key = v.starts_with('K'),
                    _ => {}
                }
            }
            Ok(meta)
        })
        .collect()
}

/// Reads decoded frames as raw planar video from `ffmpeg -f rawvideo`.
pub struct RawFrames {
    out: BufReader<Box<dyn Read + Send>>,
    process: Option<Box<dyn Process>>,
    frame_bytes: usize,
}

impl RawFrames {
    /// Decode the video stream of `input` at its own pixel format.
    ///
    /// # Errors
    /// Returns an error when ffmpeg cannot be started.
    pub fn open(launcher: &dyn Launcher, input: &Path, info: &MediaInfo) -> Result<Self> {
        let (bit_depth, sampling) = info.pixel_layout()?;
        let bytes = if bit_depth > 8 { 2 } else { 1 };
        let (cw, ch) = chroma_size(sampling, info.width, info.height);
        let frame_bytes = (info.width * info.height + 2 * cw * ch) * bytes;
        let mut args = os(&["-nostdin", "-v", "error", "-xerror", "-i"], input);
        for arg in [
            "-map",
            &format!("0:{}", info.video_index),
            "-fps_mode",
            "passthrough",
            "-f",
            "rawvideo",
            "-pix_fmt",
            &info.pix_fmt,
            "-",
        ] {
            args.push(arg.into());
        }
        let mut process = launcher.spawn(Tool::Ffmpeg, &args, false)?;
        let out = process.stdout().ok_or_else(|| anyhow!("no stdout pipe"))?;
        Ok(Self {
            out: BufReader::with_capacity(frame_bytes.max(1 << 16), out),
            process: Some(process),
            frame_bytes,
        })
    }

    /// Fill `buf` with the next frame's planes (Y, then U, then V, no padding).
    /// Returns `false` at the end of the stream.
    ///
    /// # Errors
    /// Returns an error when ffmpeg fails (e.g. a decode error) or the stream ends mid-frame.
    pub fn next_frame(&mut self, buf: &mut Vec<u8>) -> Result<bool> {
        buf.resize(self.frame_bytes, 0);
        let mut filled = 0;
        while filled < buf.len() {
            match self.out.read(&mut buf[filled..]) {
                Ok(0) => break,
                Ok(n) => filled += n,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                Err(e) => return Err(e.into()),
            }
        }
        if filled == buf.len() {
            return Ok(true);
        }
        if let Some(process) = self.process.take() {
            process.wait()?;
        }
        if filled != 0 {
            bail!("decoded video ended in the middle of a frame");
        }
        Ok(false)
    }
}

/// Chroma plane size for `sampling` at `width`×`height`.
#[must_use]
pub fn chroma_size(sampling: ChromaSubsampling, width: usize, height: usize) -> (usize, usize) {
    match sampling {
        ChromaSubsampling::Yuv420 => (width.div_ceil(2), height.div_ceil(2)),
        ChromaSubsampling::Yuv422 => (width.div_ceil(2), height),
        ChromaSubsampling::Yuv444 => (width, height),
        ChromaSubsampling::Monochrome => (0, 0),
    }
}

/// Packet timing the remux restores; see [`PacketReader::timing`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RemuxTiming {
    pub packets: usize,
    pub last_duration: i64,
    pub leading_dts: Vec<i64>,
}

/// Writes rewritten video packets to an `ffmpeg` that remuxes them with every audio, video and
/// subtitle stream, the chapters and the metadata of the original file.
pub struct Remuxer {
    nut: Option<crate::nut::NutWriter<Box<dyn Write + Send>>>,
    process: Option<Box<dyn Process>>,
}

impl Remuxer {
    /// Start `ffmpeg` writing `output`.
    ///
    /// # Errors
    /// Returns an error when ffmpeg cannot be started.
    pub fn start(
        launcher: &dyn Launcher,
        input: &Path,
        output: &Path,
        info: &MediaInfo,
        timing: &RemuxTiming,
    ) -> Result<Self> {
        let args = remux_args(input, output, info, timing);
        let process = launcher.spawn(Tool::Ffmpeg, &args, true)?;
        Ok(Self {
            nut: None,
            process: Some(process),
        })
    }

    /// Write the next video packet. `extradata` is the codec configuration for the stream
    /// header and is only used for the first packet.
    ///
    /// # Errors
    /// Returns an error when ffmpeg stopped reading (its failure is reported).
    pub fn write(&mut self, info: &MediaInfo, extradata: &[u8], packet: &Packet) -> Result<()> {
        if self.nut.is_none() {
            let stdin = self
                .process
                .as_mut()
                .and_then(|p| p.stdin())
                .ok_or_else(|| anyhow!("no stdin pipe"))?;
            let stream = crate::nut::NutStream {
                fourcc: if info.codec == "hevc" {
                    *b"HEVC"
                } else {
                    *b"av01"
                },
                time_base: info.time_base,
                decode_delay: info.has_b_frames,
                extradata: extradata.to_vec(),
                width: info.width as u64,
                height: info.height as u64,
                sample_aspect_ratio: info.sample_aspect_ratio,
            };
            let stdin: Box<dyn Write + Send> =
                Box::new(std::io::BufWriter::with_capacity(1 << 20, stdin));
            match crate::nut::NutWriter::new(stdin, &stream) {
                Ok(nut) => self.nut = Some(nut),
                Err(e) => return Err(self.fail(e.into())),
            }
        }
        let result = self.nut.as_mut().expect("set above").write_frame(
            packet.pts.unwrap_or(0),
            packet.key,
            &packet.data,
        );
        if let Err(e) = result {
            return Err(self.fail(e.into()));
        }
        Ok(())
    }

    /// Prefer ffmpeg's own failure (e.g. a muxing error) over the broken pipe it caused.
    fn fail(&mut self, error: anyhow::Error) -> anyhow::Error {
        self.nut = None;
        match self.process.take().map(|p| p.wait()) {
            Some(Err(ffmpeg)) => ffmpeg,
            _ => error,
        }
    }

    /// Close the stream and wait for ffmpeg to finish the file.
    ///
    /// # Errors
    /// Returns ffmpeg's failure.
    pub fn finish(mut self) -> Result<()> {
        if let Some(nut) = self.nut.take()
            && let Err(e) = nut.finish()
        {
            return Err(self.fail(e.into()));
        }
        if let Some(process) = self.process.take() {
            process.wait()?;
        }
        Ok(())
    }
}

/// The ffmpeg arguments that remux NUT video from stdin with the other streams of `input`.
#[must_use]
pub fn remux_args(
    input: &Path,
    output: &Path,
    info: &MediaInfo,
    timing: &RemuxTiming,
) -> Vec<OsString> {
    let mut args: Vec<OsString> = [
        "-nostdin", "-v", "error", "-y", "-copyts", "-f", "nut", "-i", "-", "-copyts", "-i",
    ]
    .iter()
    .map(OsString::from)
    .collect();
    args.push(input.into());
    let mut push = |a: &str| args.push(a.into());
    let mut out = 0;
    for (index, kind) in &info.streams {
        if !matches!(kind.as_str(), "video" | "audio" | "subtitle") {
            continue;
        }
        if *index == info.video_index {
            push("-map");
            push("0:0");
            push(&format!("-map_metadata:s:{out}"));
            push(&format!("1:s:{index}"));
            push(&format!("-disposition:{out}"));
            push(&if info.disposition.is_empty() {
                "0".to_owned()
            } else {
                info.disposition.join("+")
            });
            if info.r_frame_rate.0 > 0 && info.r_frame_rate.1 > 0 {
                // NUT stores no frame rate; containers like Matroska derive durations from it.
                // Pinning the time base stops stream copy from switching to 1/frame rate.
                push(&format!("-r:{out}"));
                push(&format!("{}/{}", info.r_frame_rate.0, info.r_frame_rate.1));
                push(&format!("-time_base:{out}"));
                push(&format!("{}/{}", info.time_base.0, info.time_base.1));
            }
            if timing.packets > 0 {
                let dts = timing
                    .leading_dts
                    .iter()
                    .enumerate()
                    .rev()
                    .fold("DTS".to_owned(), |rest, (n, dts)| {
                        format!("if(eq(N\\,{n})\\,{dts}\\,{rest})")
                    });
                push(&format!("-bsf:{out}"));
                push(&format!(
                    "setts=pts=PTS:dts={dts}:duration=if(eq(N\\,{})\\,{}\\,DURATION)",
                    timing.packets - 1,
                    timing.last_duration
                ));
            }
        } else {
            push("-map");
            push(&format!("1:{index}"));
            push(&format!("-map_metadata:s:{out}"));
            push(&format!("1:s:{index}"));
        }
        out += 1;
    }
    for a in [
        "-map_metadata",
        "1",
        "-map_chapters",
        "1",
        "-c",
        "copy",
        "-copyinkf",
    ] {
        push(a);
    }
    args.push(output.into());
    args
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_probe_output_with_extradata() {
        let text = "index=0\ncodec_name=hevc\nwidth=128\nheight=72\nhas_b_frames=2\n\
                    sample_aspect_ratio=1:1\npix_fmt=yuv420p10le\ncolor_range=tv\n\
                    color_transfer=smpte2084\nr_frame_rate=24000/1001\navg_frame_rate=0/0\n\
                    time_base=1/24000\nextradata=\n\
                    00000000: 0101 6000 0000 9000 0000 0000 1ef0 00fc  ..`.............\n\
                    00000010: fdf8                                     ..\n\
                    DISPOSITION:default=1\nDISPOSITION:dub=0\n";
        let info = parse_probe(text, vec![(0, "video".into())]).unwrap();
        assert_eq!(info.codec, "hevc");
        assert_eq!(info.frame_rate, Rational32::new_raw(24000, 1001));
        assert_eq!(info.time_base, (1, 24000));
        assert_eq!(info.sample_aspect_ratio, (1, 1));
        assert_eq!(info.extradata.len(), 18);
        assert_eq!(&info.extradata[..4], &[1, 1, 0x60, 0]);
        assert_eq!(info.extradata[16..], [0xfd, 0xf8]);
        assert_eq!(info.disposition, vec!["default".to_owned()]);
        assert_eq!(info.pixel_layout().unwrap().0, 10);
    }

    /// Answers each tool with canned stdout and an exit result.
    struct FakeLauncher {
        ffprobe: &'static str,
        ffmpeg: &'static [u8],
        ffmpeg_error: Option<&'static str>,
    }

    struct FakeProcess {
        stdout: Option<&'static [u8]>,
        error: Option<&'static str>,
    }

    impl Launcher for FakeLauncher {
        fn spawn(&self, tool: Tool, _: &[OsString], _: bool) -> Result<Box<dyn Process>> {
            Ok(Box::new(match tool {
                Tool::Ffprobe => FakeProcess {
                    stdout: Some(self.ffprobe.as_bytes()),
                    error: None,
                },
                Tool::Ffmpeg => FakeProcess {
                    stdout: Some(self.ffmpeg),
                    error: self.ffmpeg_error,
                },
            }))
        }
    }

    impl Process for FakeProcess {
        fn stdin(&mut self) -> Option<Box<dyn Write + Send>> {
            None
        }

        fn stdout(&mut self) -> Option<Box<dyn Read + Send>> {
            self.stdout
                .take()
                .map(|s| Box::new(s) as Box<dyn Read + Send>)
        }

        fn wait(self: Box<Self>) -> Result<()> {
            self.error
                .map_or(Ok(()), |e| Err(anyhow!("ffmpeg failed: {e}")))
        }
    }

    fn info_2x2() -> MediaInfo {
        parse_probe(
            "index=0\ncodec_name=av1\nwidth=2\nheight=2\npix_fmt=yuv420p\n\
             avg_frame_rate=24/1\nr_frame_rate=24/1\ntime_base=1/1000\n\
             color_transfer=unknown\ncolor_range=tv\nhas_b_frames=0\n\
             sample_aspect_ratio=1:1\n",
            vec![(0, "video".into())],
        )
        .unwrap()
    }

    const TWO_PACKETS: &str =
        "pts=0|dts=0|duration=1|size=2|flags=K__\npts=1|dts=1|duration=1|size=3|flags=___\n";

    #[test]
    fn packets_are_split_by_the_ffprobe_listing() {
        let launcher = FakeLauncher {
            ffprobe: TWO_PACKETS,
            ffmpeg: b"abcde",
            ffmpeg_error: None,
        };
        let packets: Vec<Packet> = PacketReader::open(&launcher, Path::new("x"), &info_2x2())
            .unwrap()
            .collect::<Result<_>>()
            .unwrap();
        assert_eq!(packets.len(), 2);
        assert_eq!(packets[0].data, b"ab");
        assert_eq!(packets[1].data, b"cde");
        assert_eq!(packets[1].pts, Some(1));
    }

    #[test]
    fn truncated_video_stream_is_an_error() {
        let launcher = FakeLauncher {
            ffprobe: TWO_PACKETS,
            ffmpeg: b"abc",
            ffmpeg_error: None,
        };
        let result: Result<Vec<Packet>> =
            PacketReader::open(&launcher, Path::new("x"), &info_2x2())
                .unwrap()
                .collect();
        assert!(format!("{:#}", result.unwrap_err()).contains("ended early"));
    }

    #[test]
    fn demux_failure_is_reported() {
        let launcher = FakeLauncher {
            ffprobe: TWO_PACKETS,
            ffmpeg: b"ab",
            ffmpeg_error: Some("Invalid data found when processing input"),
        };
        let result: Result<Vec<Packet>> =
            PacketReader::open(&launcher, Path::new("x"), &info_2x2())
                .unwrap()
                .collect();
        assert!(result.unwrap_err().to_string().contains("Invalid data"));
    }

    #[test]
    fn decoded_frames_are_read_whole_and_decode_errors_fail() {
        // 2x2 yuv420p: 4 luma + 2 chroma bytes per frame.
        let launcher = FakeLauncher {
            ffprobe: "",
            ffmpeg: b"123456abc",
            ffmpeg_error: Some("Error while decoding stream #0:0"),
        };
        let mut frames = RawFrames::open(&launcher, Path::new("x"), &info_2x2()).unwrap();
        let mut buf = Vec::new();
        assert!(frames.next_frame(&mut buf).unwrap());
        assert_eq!(buf, b"123456");
        let error = frames.next_frame(&mut buf).unwrap_err();
        assert!(error.to_string().contains("Error while decoding"));
    }

    #[test]
    fn decoded_stream_ending_mid_frame_is_an_error() {
        let launcher = FakeLauncher {
            ffprobe: "",
            ffmpeg: b"1234",
            ffmpeg_error: None,
        };
        let mut frames = RawFrames::open(&launcher, Path::new("x"), &info_2x2()).unwrap();
        let error = frames.next_frame(&mut Vec::new()).unwrap_err();
        assert!(error.to_string().contains("middle of a frame"));
    }

    #[test]
    fn probe_without_video_is_an_error() {
        assert!(parse_probe("", Vec::new()).is_err());
    }

    #[test]
    fn parses_packet_listing() {
        let metas = parse_packet_listing(
            "pts=0|dts=N/A|duration=33|size=1421|flags=K__\npts=100|dts=-2|duration=33|size=5|flags=___\n",
        )
        .unwrap();
        assert_eq!(metas.len(), 2);
        assert_eq!(metas[0].pts, Some(0));
        assert_eq!(metas[0].dts, None);
        assert!(metas[0].key);
        assert_eq!(metas[1].dts, Some(-2));
        assert_eq!(metas[1].size, 5);
        assert!(!metas[1].key);
    }

    #[test]
    fn remux_maps_av_and_subtitles_in_order_and_drops_data() {
        let info = MediaInfo {
            video_index: 1,
            codec: "av1".into(),
            width: 2,
            height: 2,
            pix_fmt: "yuv420p".into(),
            frame_rate: Rational32::new_raw(24, 1),
            r_frame_rate: (24, 1),
            time_base: (1, 1000),
            color_transfer: "unknown".into(),
            color_range: "tv".into(),
            has_b_frames: 0,
            sample_aspect_ratio: (0, 0),
            extradata: Vec::new(),
            disposition: vec!["default".into()],
            streams: vec![
                (0, "audio".into()),
                (1, "video".into()),
                (2, "data".into()),
                (3, "subtitle".into()),
            ],
        };
        let args: Vec<String> = remux_args(
            Path::new("in.mp4"),
            Path::new("out.mp4"),
            &info,
            &RemuxTiming {
                packets: 3,
                last_duration: 7,
                leading_dts: vec![-2, -1],
            },
        )
        .into_iter()
        .map(|a| a.into_string().unwrap())
        .collect();
        let joined = args.join(" ");
        assert!(joined.contains(
            "-map 1:0 -map_metadata:s:0 1:s:0 -map 0:0 -map_metadata:s:1 1:s:1 -disposition:1 default -r:1 24/1 -time_base:1 1/1000 -bsf:1 setts=pts=PTS:dts=if(eq(N\\,0)\\,-2\\,if(eq(N\\,1)\\,-1\\,DTS)):duration=if(eq(N\\,2)\\,7\\,DURATION) -map 1:3 -map_metadata:s:2 1:s:3 -map_metadata 1"
        ));
        assert!(!joined.contains("1:2 "));
        assert_eq!(args.last().unwrap(), "out.mp4");
    }
}

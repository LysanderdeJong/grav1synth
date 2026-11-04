use anyhow::{Result, anyhow};
use log::{debug, log_enabled, warn};
use nom::Finish;

use self::{
    frame::{FrameHeader, NUM_REF_FRAMES, REFS_PER_FRAME, RefType},
    grain::FilmGrainHeader,
    obu::Obu,
    sequence::SequenceHeader,
};
use crate::{GrainTableSegment, tools::Packet};

pub mod frame;
pub mod grain;
pub mod obu;
pub mod sequence;
pub mod tile_group;
pub mod trace;
pub mod util;

/// AV1 bitstream parser. With `WRITE` it also rewrites each packet's film grain headers.
///
/// It sees only packet payloads; the caller does the I/O (see [`crate::tools`]).
pub struct BitstreamParser<const WRITE: bool> {
    packet_out: Vec<u8>,
    incoming_grain_header: Option<Vec<GrainTableSegment>>,
    parsed: bool,
    size: usize,
    seen_frame_header: bool,
    sequence_header: Option<SequenceHeader>,
    previous_frame_header: Option<FrameHeader>,
    ref_frame_idx: [usize; REFS_PER_FRAME],
    ref_order_hint: [u64; NUM_REF_FRAMES],
    big_ref_order_hint: [u64; NUM_REF_FRAMES],
    big_ref_valid: [bool; NUM_REF_FRAMES],
    big_order_hints: [u64; RefType::Last as usize + REFS_PER_FRAME],
    grain_headers: Vec<FilmGrainHeader>,
}

impl<const WRITE: bool> Default for BitstreamParser<WRITE> {
    fn default() -> Self {
        Self {
            packet_out: Vec::new(),
            incoming_grain_header: None,
            parsed: false,
            size: 0,
            seen_frame_header: false,
            sequence_header: None,
            previous_frame_header: None,
            ref_frame_idx: Default::default(),
            ref_order_hint: Default::default(),
            big_ref_order_hint: Default::default(),
            big_ref_valid: Default::default(),
            big_order_hints: Default::default(),
            grain_headers: Vec::new(),
        }
    }
}

impl BitstreamParser<false> {
    /// A read-only parser.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl BitstreamParser<true> {
    /// A parser that rewrites grain headers to `incoming_grain_header` (`None` removes grain).
    #[must_use]
    pub fn with_grain(incoming_grain_header: Option<Vec<GrainTableSegment>>) -> Self {
        Self {
            incoming_grain_header,
            ..Self::default()
        }
    }
}

/// Convert a packet pts in `time_base` (num, den) to 100ns grain table units.
#[must_use]
pub fn pts_to_grain_ts(pts: i64, (num, den): (u64, u64)) -> u64 {
    if pts < 0 || den == 0 {
        return 0;
    }
    // Use ceiling so timestamps line up with the 100ns packet boundaries that
    // are emitted by `aggregate_grain_headers` when generating the grain table.
    (pts as u64 * num * 10_000_000u64).div_ceil(den)
}

/// Maps packet pts to grain table time the way libaom does (the frame's presentation time),
/// undoing container rounding.
///
/// Containers with a coarse time base (Matroska: 1 ms) store each frame time rounded, so frame 2
/// at 24 fps reads as 83 ms instead of 83.33 ms and would fall into the table segment before
/// the one libaom picks. When a pts is within half a time-base unit of a frame time on the
/// `frame_rate` grid (anchored at the first pts), the exact frame time is used. Other pts (VFR)
/// are used as they are.
#[derive(Debug, Clone)]
pub struct FrameClock {
    time_base: (u64, u64),
    frame_rate: (u64, u64),
    start: Option<i64>,
}

impl FrameClock {
    /// `time_base` and `frame_rate` as (num, den); a zero `frame_rate` disables snapping.
    #[must_use]
    pub const fn new(time_base: (u64, u64), frame_rate: (u64, u64)) -> Self {
        Self {
            time_base,
            frame_rate,
            start: None,
        }
    }

    /// Grain table time (100ns units, rounded up) of the packet with `pts`.
    pub fn grain_ts(&mut self, pts: i64) -> u64 {
        let ((tb_num, tb_den), (fps_num, fps_den)) = (self.time_base, self.frame_rate);
        let start = *self.start.get_or_insert(pts);
        if pts < 0 || tb_den == 0 || fps_num == 0 || fps_den == 0 {
            return pts_to_grain_ts(pts, self.time_base);
        }
        // Times in units of 1 / (tb_den * fps_num) seconds.
        let (tb_num, tb_den, fps_num, fps_den) = (
            i128::from(tb_num),
            i128::from(tb_den),
            i128::from(fps_num),
            i128::from(fps_den),
        );
        let time = i128::from(pts) * tb_num * fps_num;
        let origin = i128::from(start) * tb_num * fps_num;
        let frame = fps_den * tb_den;
        let index = (2 * (time - origin) + frame).div_euclid(2 * frame);
        let on_grid = origin + index * frame;
        let time = if 2 * (time - on_grid).abs() <= tb_num * fps_num {
            on_grid
        } else {
            time
        };
        if time <= 0 {
            return 0;
        }
        let unit = tb_den * fps_num;
        let ticks = (time * 10_000_000 + unit - 1) / unit;
        u64::try_from(ticks).unwrap_or(u64::MAX)
    }
}

impl<const WRITE: bool> BitstreamParser<WRITE> {
    /// Parse every OBU of one packet. `collect` records each frame's grain header.
    fn parse_packet(&mut self, mut input: &[u8], packet_ts: u64, collect: bool) -> Result<()> {
        loop {
            let (remaining, obu) = self
                .parse_obu(input, packet_ts)
                .finish()
                .map_err(|e| anyhow!("{e:?}"))?;
            input = remaining;
            match obu {
                Some(Obu::SequenceHeader(obu)) => {
                    self.sequence_header = Some(obu);
                }
                Some(Obu::FrameHeader(obu)) => {
                    if collect {
                        self.grain_headers.push(obu.film_grain_params.clone());
                    }
                    self.previous_frame_header = Some(obu);
                }
                None => (),
            }
            if input.is_empty() {
                return Ok(());
            }
        }
    }

    /// Returns `true` if the `film_grain_params_present` flag is set in the stream's Sequence
    /// Header, and `false` otherwise.
    ///
    /// This reads only as far as the first Sequence Header OBU — typically the very start of
    /// the first video packet — so it is far faster than [`Self::get_grain_headers`] for a
    /// simple presence check.
    ///
    /// # Errors
    /// Returns packet read and parse errors.
    pub fn film_grain_params_present(
        &mut self,
        packets: impl IntoIterator<Item = Result<Packet>>,
    ) -> Result<bool> {
        if self.sequence_header.is_none() {
            'packets: for packet in packets {
                let packet = packet?;
                let mut input = &packet.data[..];
                while !input.is_empty() {
                    let (remaining, obu) = self
                        .parse_obu(input, 0)
                        .finish()
                        .map_err(|e| anyhow!("{e:?}"))?;
                    input = remaining;
                    if let Some(Obu::SequenceHeader(sh)) = obu {
                        self.sequence_header = Some(sh);
                        break 'packets;
                    }
                }
            }
        }

        Ok(self
            .sequence_header
            .as_ref()
            .is_some_and(|sh| sh.film_grain_params_present))
    }

    /// Parse every packet and return each frame's film grain header.
    ///
    /// # Errors
    /// Returns packet read and parse errors.
    pub fn get_grain_headers(
        &mut self,
        packets: impl IntoIterator<Item = Result<Packet>>,
        time_base: (u64, u64),
    ) -> Result<&[FilmGrainHeader]> {
        if self.parsed {
            return Ok(&self.grain_headers);
        }

        for packet in packets {
            let packet = packet?;
            debug!(
                target: "trace_headers",
                "Packet: {} bytes, pts {}, dts {}.",
                packet.data.len(),
                packet.pts.unwrap_or_default(),
                packet.dts.unwrap_or_default(),
            );
            let packet_ts = pts_to_grain_ts(packet.pts.unwrap_or_default(), time_base);
            self.parse_packet(&packet.data, packet_ts, true)?;
        }

        self.parsed = true;

        Ok(&self.grain_headers)
    }

    /// The sequence header seen most recently, if any.
    #[must_use]
    pub const fn sequence_header(&self) -> Option<&SequenceHeader> {
        self.sequence_header.as_ref()
    }

    /// Rewrite one packet's grain headers and return the new payload.
    ///
    /// # Errors
    /// Returns parse errors.
    ///
    /// # Panics
    /// Panics when `WRITE` is false.
    pub fn modify_packet(&mut self, data: &[u8], packet_ts: u64) -> Result<Vec<u8>> {
        assert!(
            WRITE,
            "Can only modify headers if the WRITE generic is true"
        );

        debug!(
            target: "trace_headers",
            "Packet: {} bytes, ts {packet_ts}.",
            data.len(),
        );
        self.parse_packet(data, packet_ts, false)?;
        let out = std::mem::take(&mut self.packet_out);
        if out.len() != data.len() {
            debug!("Resized packet from {} to {}", data.len(), out.len());
        }
        if log_enabled!(target: "trace_headers", log::Level::Debug) {
            self.debug_reparse(&out, packet_ts);
        }
        Ok(out)
    }

    fn debug_reparse(&self, data: &[u8], packet_ts: u64) {
        debug!(
            target: "trace_headers",
            "=== Re-parsing modified packet: {} bytes ===",
            data.len(),
        );
        let mut read_parser = BitstreamParser::<false> {
            size: self.size,
            seen_frame_header: self.seen_frame_header,
            sequence_header: self.sequence_header.clone(),
            previous_frame_header: self.previous_frame_header.clone(),
            ref_frame_idx: self.ref_frame_idx,
            ref_order_hint: self.ref_order_hint,
            big_ref_order_hint: self.big_ref_order_hint,
            big_ref_valid: self.big_ref_valid,
            big_order_hints: self.big_order_hints,
            ..BitstreamParser::default()
        };
        let mut input = data;
        while !input.is_empty() {
            match read_parser.parse_obu(input, packet_ts).finish() {
                Ok((remaining, _)) => input = remaining,
                Err(e) => {
                    warn!("Debug re-parse of modified packet failed: {e:?}");
                    break;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_parsed_parser<const WRITE: bool>(
        headers: Vec<FilmGrainHeader>,
    ) -> BitstreamParser<WRITE> {
        BitstreamParser {
            parsed: true,
            grain_headers: headers,
            ..BitstreamParser::default()
        }
    }

    fn sample_grain_params() -> grain::FilmGrainParams {
        grain::FilmGrainParams {
            grain_seed: 42,
            scaling_points_y: Default::default(),
            scaling_points_cb: Default::default(),
            scaling_points_cr: Default::default(),
            scaling_shift: 8,
            ar_coeff_lag: 0,
            ar_coeffs_y: Default::default(),
            ar_coeffs_cb: Default::default(),
            ar_coeffs_cr: Default::default(),
            ar_coeff_shift: 6,
            cb_mult: 0,
            cb_luma_mult: 0,
            cb_offset: 0,
            cr_mult: 0,
            cr_luma_mult: 0,
            cr_offset: 0,
            chroma_scaling_from_luma: false,
            grain_scale_shift: 0,
            overlap_flag: false,
            clip_to_restricted_range: false,
        }
    }

    fn no_packets() -> std::iter::Empty<Result<Packet>> {
        std::iter::empty()
    }

    #[test]
    fn get_grain_headers_returns_cached_when_already_parsed() {
        let headers = vec![
            FilmGrainHeader::Disable,
            FilmGrainHeader::UpdateGrain(sample_grain_params()),
        ];
        let mut parser = make_parsed_parser::<false>(headers);

        let result = parser
            .get_grain_headers(no_packets(), (1, 1000))
            .expect("should return cached headers");

        assert_eq!(result.len(), 2);
        assert_eq!(result[0], FilmGrainHeader::Disable);
    }

    #[test]
    fn get_grain_headers_returns_empty_without_packets() {
        let mut parser = BitstreamParser::new();

        let result = parser
            .get_grain_headers(no_packets(), (1, 1000))
            .expect("should return empty");

        assert!(result.is_empty());
        assert!(parser.parsed);
    }

    #[test]
    fn get_grain_headers_preserves_all_grain_variants() {
        let headers = vec![
            FilmGrainHeader::Disable,
            FilmGrainHeader::CopyRefFrame,
            FilmGrainHeader::UpdateGrain(sample_grain_params()),
        ];
        let mut parser = make_parsed_parser::<false>(headers);

        let result = parser
            .get_grain_headers(no_packets(), (1, 1000))
            .expect("should preserve variants");

        assert_eq!(result.len(), 3);
        assert_eq!(result[0], FilmGrainHeader::Disable);
        assert_eq!(result[1], FilmGrainHeader::CopyRefFrame);
        assert!(matches!(result[2], FilmGrainHeader::UpdateGrain(_)));
    }

    #[test]
    fn get_grain_headers_propagates_packet_errors() {
        let mut parser = BitstreamParser::new();

        let result = parser.get_grain_headers([Err(anyhow!("truncated input"))], (1, 1000));

        assert!(result.unwrap_err().to_string().contains("truncated"));
    }

    #[test]
    fn film_grain_params_present_is_false_without_sequence_header() {
        let mut parser = BitstreamParser::new();
        assert!(!parser.film_grain_params_present(no_packets()).unwrap());
    }

    #[test]
    #[should_panic(expected = "Can only modify headers")]
    fn modify_packet_panics_when_write_is_false() {
        let mut parser = BitstreamParser::new();
        let _ = parser.modify_packet(&[], 0);
    }

    #[test]
    fn pts_to_grain_ts_rounds_up_to_100ns() {
        assert_eq!(pts_to_grain_ts(1, (1, 90_000)), 112);
        assert_eq!(pts_to_grain_ts(1001, (1, 24_000)), 417_084);
        assert_eq!(pts_to_grain_ts(-5, (1, 1000)), 0);
        assert_eq!(pts_to_grain_ts(5, (1, 0)), 0);
    }

    #[test]
    fn frame_clock_undoes_millisecond_rounding_on_the_frame_grid() {
        let mut clock = FrameClock::new((1, 1000), (24, 1));
        // 0, 41.67, 83.33, 125 ms stored as 0, 42, 83, 125.
        let ticks: Vec<u64> = [0, 42, 83, 125].map(|pts| clock.grain_ts(pts)).to_vec();
        assert_eq!(ticks, [0, 416_667, 833_334, 1_250_000]);
        // Exactly what frame index / frame rate gives, as libaom sees it.
        assert!(
            ticks
                .iter()
                .enumerate()
                .all(|(n, &t)| t == (n as u64 * 10_000_000).div_ceil(24))
        );
    }

    #[test]
    fn frame_clock_anchors_the_grid_at_the_first_pts() {
        // 23.976 fps starting at 10 s in a 1 ms time base: 10000, 10042, 10083 ms.
        let mut clock = FrameClock::new((1, 1000), (24_000, 1001));
        assert_eq!(clock.grain_ts(10_000), 100_000_000);
        assert_eq!(clock.grain_ts(10_042), 100_417_084);
        assert_eq!(clock.grain_ts(10_083), 100_834_167);
    }

    #[test]
    fn frame_clock_keeps_off_grid_and_exact_timestamps() {
        // VFR: 60 ms is not within 0.5 ms of a 24 fps frame time.
        let mut clock = FrameClock::new((1, 1000), (24, 1));
        assert_eq!(clock.grain_ts(0), 0);
        assert_eq!(clock.grain_ts(60), 600_000);
        // A fine time base already holds the exact time.
        let mut clock = FrameClock::new((1, 90_000), (24, 1));
        assert_eq!(clock.grain_ts(3750), pts_to_grain_ts(3750, (1, 90_000)));
        // No frame rate: plain conversion.
        let mut clock = FrameClock::new((1, 1000), (0, 0));
        assert_eq!(clock.grain_ts(83), 830_000);
    }

    #[cfg(feature = "dav1d_tests")]
    mod io {
        use super::*;
        use crate::tools::{PacketReader, StdLauncher, probe};
        use std::path::PathBuf;

        fn packets(relative: &str) -> (PacketReader, (u64, u64)) {
            let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("dav1d-test-data")
                .join(relative);
            let info = probe(&StdLauncher, &path).expect("test file should probe");
            let reader = PacketReader::open(&StdLauncher, &path, &info).expect("should open");
            (reader, info.time_base)
        }

        #[test]
        fn get_grain_headers_parses_valid_file() {
            let (reader, tb) = packets("8-bit/data/00000000.ivf");
            let mut parser = BitstreamParser::new();

            let result = parser.get_grain_headers(reader, tb).expect("should parse");

            assert!(!result.is_empty());
            assert!(parser.sequence_header.is_some());
            assert!(parser.previous_frame_header.is_some());
        }

        #[test]
        fn modify_packet_rewrites_every_packet() {
            let (reader, tb) = packets("8-bit/data/00000000.ivf");
            let mut parser = BitstreamParser::with_grain(None);

            for packet in reader {
                let packet = packet.expect("packet");
                let ts = pts_to_grain_ts(packet.pts.unwrap_or_default(), tb);
                let out = parser.modify_packet(&packet.data, ts).expect("rewrite");
                assert!(!out.is_empty());
            }
            assert!(
                parser.grain_headers.is_empty(),
                "write path should not collect grain headers"
            );
        }
    }
}

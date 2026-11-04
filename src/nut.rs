//! Minimal NUT muxer for one video stream, read back by `ffmpeg -f nut -i -`.
//!
//! NUT keeps any rational time base and every packet's pts exactly, carries the codec
//! configuration (extradata), and lets the demuxer rebuild B-frame dts from the stream's
//! decode delay. It stores no packet durations. Layout follows FFmpeg's `nutdec.c`: version 4
//! with the pipe flag (one syncpoint, no index), one frame code with explicit flags, and a
//! checksum on every frame header.

use std::io::{self, Write};

const ID_STRING: &[u8] = b"nut/multimedia container\0";
const fn startcode(tag: &[u8; 2], low: u64) -> u64 {
    low + (((tag[0] as u64) << 8 | tag[1] as u64) << 48)
}
const MAIN_STARTCODE: u64 = startcode(b"NM", 0x7A56_1F5F_04AD);
const STREAM_STARTCODE: u64 = startcode(b"NS", 0x1140_5BF2_F9DB);
const SYNCPOINT_STARTCODE: u64 = startcode(b"NK", 0xE4AD_EECA_4569);

const FLAG_KEY: u64 = 1;
const FLAG_CODED_PTS: u64 = 8;
const FLAG_SIZE_MSB: u64 = 32;
const FLAG_CHECKSUM: u64 = 64;
const FLAG_CODED: u64 = 4096;
const NUT_PIPE: u64 = 2;
const MSB_PTS_SHIFT: u32 = 15;

/// The video stream written to the file.
#[derive(Debug, Clone)]
pub struct NutStream {
    pub fourcc: [u8; 4],
    /// (num, den); reduced before writing.
    pub time_base: (u64, u64),
    /// Frames of reordering delay (ffprobe `has_b_frames`).
    pub decode_delay: u64,
    pub extradata: Vec<u8>,
    pub width: u64,
    pub height: u64,
    /// (0, 0) when unknown.
    pub sample_aspect_ratio: (u64, u64),
}

/// Writes NUT frames for a single stream.
pub struct NutWriter<W: Write> {
    out: W,
    last_pts: Option<i64>,
}

impl<W: Write> NutWriter<W> {
    /// Write the file and stream headers.
    ///
    /// # Errors
    /// Returns write errors.
    pub fn new(mut out: W, stream: &NutStream) -> io::Result<Self> {
        out.write_all(ID_STRING)?;

        let (num, den) = reduce(stream.time_base);
        let mut main = Vec::new();
        put_v(&mut main, 4); // version
        put_v(&mut main, 1); // minor version
        put_v(&mut main, 1); // stream count
        put_v(&mut main, 65536); // max distance
        put_v(&mut main, 1); // time base count
        put_v(&mut main, num);
        put_v(&mut main, den);
        // One frame-code group covering all 255 usable codes: every field is coded per frame.
        put_v(&mut main, FLAG_CODED);
        put_v(&mut main, 6); // fields: pts, mul, stream, size, reserved, count
        put_v(&mut main, 0); // pts delta (signed 0)
        put_v(&mut main, 1); // size mul
        put_v(&mut main, 0); // stream
        put_v(&mut main, 0); // size lsb
        put_v(&mut main, 0); // reserved count
        put_v(&mut main, 255); // count (256 minus the reserved 'N')
        put_v(&mut main, 0); // header count minus 1: no elision headers
        put_v(&mut main, NUT_PIPE);
        put_packet(&mut out, MAIN_STARTCODE, &main)?;

        let mut header = Vec::new();
        put_v(&mut header, 0); // stream id
        put_v(&mut header, 0); // class: video
        put_v(&mut header, 4);
        header.extend_from_slice(&stream.fourcc);
        put_v(&mut header, 0); // time base id
        put_v(&mut header, u64::from(MSB_PTS_SHIFT));
        put_v(&mut header, 65536); // max pts distance (frames carry checksums anyway)
        put_v(&mut header, stream.decode_delay);
        put_v(&mut header, 0); // stream flags
        put_v(&mut header, stream.extradata.len() as u64);
        header.extend_from_slice(&stream.extradata);
        put_v(&mut header, stream.width);
        put_v(&mut header, stream.height);
        let (sar_num, sar_den) = stream.sample_aspect_ratio;
        let (sar_num, sar_den) = if sar_num == 0 || sar_den == 0 {
            (0, 0)
        } else {
            (sar_num, sar_den)
        };
        put_v(&mut header, sar_num);
        put_v(&mut header, sar_den);
        put_v(&mut header, 0); // colourspace type: unknown
        put_packet(&mut out, STREAM_STARTCODE, &header)?;

        Ok(Self {
            out,
            last_pts: None,
        })
    }

    /// Write one frame.
    ///
    /// # Errors
    /// Returns write errors, or `InvalidInput` for a negative pts that is too far from the
    /// previous one to be coded.
    pub fn write_frame(&mut self, pts: i64, key: bool, data: &[u8]) -> io::Result<()> {
        let last_pts = match self.last_pts {
            Some(last) => last,
            None => {
                // The syncpoint's timestamp is unsigned; it only seeds the lsb-coded pts below.
                let start = pts.max(0);
                let mut sync = Vec::new();
                put_v(&mut sync, start as u64); // global key pts (time base 0)
                put_v(&mut sync, 0); // back pointer
                put_packet(&mut self.out, SYNCPOINT_STARTCODE, &sync)?;
                start
            }
        };
        let coded_pts = if pts >= 0 {
            pts as u64 + (1 << MSB_PTS_SHIFT)
        } else {
            // Negative pts are only reachable relative to the previous pts.
            let half = (1i64 << MSB_PTS_SHIFT) / 2 - 1;
            if (pts - last_pts).abs() >= half {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("negative pts {pts} too far from {last_pts} for NUT"),
                ));
            }
            (pts as u64) & ((1 << MSB_PTS_SHIFT) - 1)
        };

        let mut header = vec![0u8]; // frame code 0
        let flags = FLAG_CODED_PTS | FLAG_SIZE_MSB | FLAG_CHECKSUM | if key { FLAG_KEY } else { 0 };
        put_v(&mut header, flags);
        put_v(&mut header, coded_pts);
        put_v(&mut header, data.len() as u64);
        let checksum = crc(0, &header);
        header.extend_from_slice(&checksum.to_be_bytes());
        self.out.write_all(&header)?;
        self.out.write_all(data)?;
        self.last_pts = Some(pts);
        Ok(())
    }

    /// Flush and close.
    ///
    /// # Errors
    /// Returns write errors.
    pub fn finish(mut self) -> io::Result<()> {
        self.out.flush()
    }
}

fn reduce((num, den): (u64, u64)) -> (u64, u64) {
    let (mut a, mut b) = (num, den);
    while b != 0 {
        (a, b) = (b, a % b);
    }
    match (num.checked_div(a), den.checked_div(a)) {
        (Some(num), Some(den)) => (num, den),
        _ => (num, den),
    }
}

fn put_v(out: &mut Vec<u8>, value: u64) {
    let mut groups = 1;
    while groups < 10 && value >> (7 * groups) != 0 {
        groups += 1;
    }
    for i in (0..groups).rev() {
        let byte = ((value >> (7 * i)) & 0x7f) as u8;
        out.push(if i == 0 { byte } else { byte | 0x80 });
    }
}

fn put_packet(out: &mut impl Write, startcode: u64, data: &[u8]) -> io::Result<()> {
    let mut header = startcode.to_be_bytes().to_vec();
    let forward_ptr = data.len() as u64 + 4;
    put_v(&mut header, forward_ptr);
    if forward_ptr > 4096 {
        let checksum = crc(0, &header);
        header.extend_from_slice(&checksum.to_be_bytes());
    }
    out.write_all(&header)?;
    out.write_all(data)?;
    out.write_all(&crc(0, data).to_be_bytes())
}

/// CRC-32 with generator 0x04C11DB7, MSB first, no reflection or final xor (FFmpeg's
/// `ff_crc04C11DB7_update`).
fn crc(mut crc: u32, data: &[u8]) -> u32 {
    for &byte in data {
        crc ^= u32::from(byte) << 24;
        for _ in 0..8 {
            crc = if crc & 0x8000_0000 != 0 {
                (crc << 1) ^ 0x04C1_1DB7
            } else {
                crc << 1
            };
        }
    }
    crc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn varints_match_nut_encoding() {
        let mut out = Vec::new();
        put_v(&mut out, 0);
        put_v(&mut out, 127);
        put_v(&mut out, 128);
        put_v(&mut out, 65536);
        assert_eq!(out, [0x00, 0x7f, 0x81, 0x00, 0x84, 0x80, 0x00]);
    }

    #[test]
    fn crc_matches_known_value() {
        // CRC-32/MPEG-2 check value is 0x0376E6E7 with init 0xFFFFFFFF; with init 0 the
        // residue property must hold: crc(data ++ be(crc(data))) == 0.
        let data = b"123456789";
        let c = crc(0, data);
        let mut with = data.to_vec();
        with.extend_from_slice(&c.to_be_bytes());
        assert_eq!(crc(0, &with), 0);
        assert_eq!(crc(0xFFFF_FFFF, data), 0x0376_E6E7);
    }

    #[test]
    fn time_base_is_reduced() {
        assert_eq!(reduce((2, 48000)), (1, 24000));
        assert_eq!(reduce((1001, 30000)), (1001, 30000));
    }
}

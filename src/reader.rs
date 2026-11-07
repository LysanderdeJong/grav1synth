use std::{num::NonZeroUsize, path::Path};

use anyhow::Result;
use av1_grain::v_frame::{
    chroma::ChromaSubsampling,
    frame::{Frame, FrameBuilder},
    pixel::Pixel,
    plane::Plane,
};
use num_rational::Rational32;
use rayon::scope;

use crate::{
    profile::{self, MetricId},
    tools::{Launcher, MediaInfo, RawFrames, chroma_size, probe},
};

/// Decoded frames of a file's video stream, read from `ffmpeg -f rawvideo`.
pub struct BitstreamReader {
    frames: RawFrames,
    video_details: VideoDetails,
    info: MediaInfo,
    buf: Vec<u8>,
}

#[derive(Debug, Clone, Copy)]
pub struct VideoDetails {
    pub width: usize,
    pub height: usize,
    pub bit_depth: usize,
    pub chroma_sampling: ChromaSubsampling,
    pub frame_rate: Rational32,
}

/// One raw frame split into its planes.
struct Planes<'a> {
    data: [&'a [u8]; 3],
    stride: [usize; 3],
}

impl<'a> Planes<'a> {
    fn split(details: &VideoDetails, buf: &'a [u8]) -> Self {
        let bytes = if details.bit_depth > 8 { 2 } else { 1 };
        let (cw, ch) = chroma_size(details.chroma_sampling, details.width, details.height);
        let y_len = details.width * details.height * bytes;
        let c_len = cw * ch * bytes;
        Self {
            data: [
                &buf[..y_len],
                &buf[y_len..y_len + c_len],
                &buf[y_len + c_len..y_len + 2 * c_len],
            ],
            stride: [details.width * bytes, cw * bytes, cw * bytes],
        }
    }

    const fn data(&self, plane: usize) -> &'a [u8] {
        self.data[plane]
    }

    const fn stride(&self, plane: usize) -> usize {
        self.stride[plane]
    }
}

impl BitstreamReader {
    /// Probe `input` and start decoding its video stream.
    ///
    /// # Errors
    /// Returns probe errors, unsupported pixel formats, and ffmpeg start failures.
    pub fn open<P: AsRef<Path>>(launcher: &dyn Launcher, input: P) -> Result<Self> {
        let info = probe(launcher, input.as_ref())?;
        let (bit_depth, chroma_sampling) = info.pixel_layout()?;
        let frames = RawFrames::open(launcher, input.as_ref(), &info)?;
        Ok(Self {
            frames,
            video_details: VideoDetails {
                width: info.width,
                height: info.height,
                bit_depth,
                chroma_sampling,
                frame_rate: info.frame_rate,
            },
            info,
            buf: Vec::new(),
        })
    }

    #[must_use]
    pub const fn get_video_details(&self) -> &VideoDetails {
        &self.video_details
    }

    /// What ffprobe reported about the file.
    #[must_use]
    pub const fn info(&self) -> &MediaInfo {
        &self.info
    }

    /// The next frame at its native bit depth, or `None` at the end.
    ///
    /// # Errors
    /// Returns decode (ffmpeg) failures.
    pub fn get_frame<T: Pixel>(&mut self) -> Result<Option<Frame<T>>> {
        if !profile::time(MetricId::ReaderReceiveFrame, || {
            self.frames.next_frame(&mut self.buf)
        })? {
            return Ok(None);
        }
        let planes = Planes::split(&self.video_details, &self.buf);
        decode_frame::<T>(&self.video_details, &planes).map(Some)
    }

    /// The next frame reduced to 8 bits, or `None` at the end.
    ///
    /// # Errors
    /// Returns decode (ffmpeg) failures.
    pub fn get_frame_u8(&mut self) -> Result<Option<Frame<u8>>> {
        if !profile::time(MetricId::ReaderReceiveFrame, || {
            self.frames.next_frame(&mut self.buf)
        })? {
            return Ok(None);
        }
        let planes = Planes::split(&self.video_details, &self.buf);
        decode_frame_u8(&self.video_details, &planes).map(Some)
    }
}

fn decode_frame<T: Pixel>(details: &VideoDetails, decoded: &Planes<'_>) -> Result<Frame<T>> {
    let width = details.width;
    let height = details.height;
    let bit_depth = details.bit_depth as u8;

    NonZeroUsize::new(width)
        .ok_or_else(|| anyhow::anyhow!("zero-width resolution is not supported"))?;
    NonZeroUsize::new(height)
        .ok_or_else(|| anyhow::anyhow!("zero-height resolution is not supported"))?;
    std::num::NonZeroU8::new(bit_depth)
        .ok_or_else(|| anyhow::anyhow!("zero bit-depth is not supported"))?;

    let mut frame: Frame<T> = profile::time(MetricId::ReaderBuildFrame, || {
        FrameBuilder::new(width, height, details.chroma_sampling, bit_depth)
            .build()
            .map_err(|e| anyhow::anyhow!("{e}"))
    })?;

    let y_stride = decoded.stride(0);
    NonZeroUsize::new(y_stride).ok_or_else(|| anyhow::anyhow!("luma stride is zero"))?;
    if let (Some(u_plane), Some(v_plane)) = (frame.u_plane.as_mut(), frame.v_plane.as_mut()) {
        let u_stride = decoded.stride(1);
        NonZeroUsize::new(u_stride).ok_or_else(|| anyhow::anyhow!("U chroma stride is zero"))?;
        let v_stride = decoded.stride(2);
        NonZeroUsize::new(v_stride).ok_or_else(|| anyhow::anyhow!("V chroma stride is zero"))?;
        let mut y_result = Ok(());
        let mut u_result = Ok(());
        let mut v_result = Ok(());
        scope(|s| {
            s.spawn(|_| {
                y_result = profile::time(MetricId::ReaderCopyY, || {
                    frame
                        .y_plane
                        .copy_from_u8_slice_with_stride(decoded.data(0), y_stride)
                        .map_err(|e| anyhow::anyhow!("luma plane copy failed: {e}"))
                });
            });
            s.spawn(|_| {
                u_result = profile::time(MetricId::ReaderCopyU, || {
                    u_plane
                        .copy_from_u8_slice_with_stride(decoded.data(1), u_stride)
                        .map_err(|e| anyhow::anyhow!("U chroma plane copy failed: {e}"))
                });
            });
            s.spawn(|_| {
                v_result = profile::time(MetricId::ReaderCopyV, || {
                    v_plane
                        .copy_from_u8_slice_with_stride(decoded.data(2), v_stride)
                        .map_err(|e| anyhow::anyhow!("V chroma plane copy failed: {e}"))
                });
            });
        });
        y_result?;
        u_result?;
        v_result?;
    } else {
        profile::time(MetricId::ReaderCopyY, || {
            frame
                .y_plane
                .copy_from_u8_slice_with_stride(decoded.data(0), y_stride)
                .map_err(|e| anyhow::anyhow!("luma plane copy failed: {e}"))
        })?;
    }

    Ok(frame)
}

fn decode_frame_u8(details: &VideoDetails, decoded: &Planes<'_>) -> Result<Frame<u8>> {
    let width = details.width;
    let height = details.height;

    NonZeroUsize::new(width)
        .ok_or_else(|| anyhow::anyhow!("zero-width resolution is not supported"))?;
    NonZeroUsize::new(height)
        .ok_or_else(|| anyhow::anyhow!("zero-height resolution is not supported"))?;

    let mut frame: Frame<u8> = profile::time(MetricId::ReaderBuildFrame, || {
        FrameBuilder::new(width, height, details.chroma_sampling, 8)
            .build()
            .map_err(|e| anyhow::anyhow!("{e}"))
    })?;

    let y_stride = decoded.stride(0);
    NonZeroUsize::new(y_stride).ok_or_else(|| anyhow::anyhow!("luma stride is zero"))?;
    let shift = details.bit_depth.saturating_sub(8);
    if let (Some(u_plane), Some(v_plane)) = (frame.u_plane.as_mut(), frame.v_plane.as_mut()) {
        let u_stride = decoded.stride(1);
        NonZeroUsize::new(u_stride).ok_or_else(|| anyhow::anyhow!("U chroma stride is zero"))?;
        let v_stride = decoded.stride(2);
        NonZeroUsize::new(v_stride).ok_or_else(|| anyhow::anyhow!("V chroma stride is zero"))?;
        let mut y_result = Ok(());
        let mut u_result = Ok(());
        let mut v_result = Ok(());
        scope(|s| {
            s.spawn(|_| {
                y_result = profile::time(MetricId::ReaderCopyY, || {
                    copy_decoded_plane_to_u8(decoded.data(0), y_stride, &mut frame.y_plane, shift)
                        .map_err(|e| anyhow::anyhow!("luma plane copy failed: {e}"))
                });
            });
            s.spawn(|_| {
                u_result = profile::time(MetricId::ReaderCopyU, || {
                    copy_decoded_plane_to_u8(decoded.data(1), u_stride, u_plane, shift)
                        .map_err(|e| anyhow::anyhow!("U chroma plane copy failed: {e}"))
                });
            });
            s.spawn(|_| {
                v_result = profile::time(MetricId::ReaderCopyV, || {
                    copy_decoded_plane_to_u8(decoded.data(2), v_stride, v_plane, shift)
                        .map_err(|e| anyhow::anyhow!("V chroma plane copy failed: {e}"))
                });
            });
        });
        y_result?;
        u_result?;
        v_result?;
    } else {
        profile::time(MetricId::ReaderCopyY, || {
            copy_decoded_plane_to_u8(decoded.data(0), y_stride, &mut frame.y_plane, shift)
                .map_err(|e| anyhow::anyhow!("luma plane copy failed: {e}"))
        })?;
    }

    Ok(frame)
}

fn copy_decoded_plane_to_u8(
    src: &[u8],
    input_stride: usize,
    out_plane: &mut Plane<u8>,
    shift: usize,
) -> Result<(), av1_grain::v_frame::plane::CopyError> {
    if shift == 0 {
        return out_plane.copy_from_u8_slice_with_stride(src, input_stride);
    }

    let width = out_plane.width();
    let row_byte_width = width * 2;
    let byte_count = input_stride * out_plane.height();
    if byte_count != src.len() {
        return Err(av1_grain::v_frame::plane::CopyError::DataLength {
            expected: byte_count,
            found: src.len(),
        });
    }

    let half = (1u16 << shift) >> 1;
    for (row_idx, out_row) in out_plane.rows_mut().enumerate() {
        let src_offset = row_idx * input_stride;
        let src_row = &src[src_offset..src_offset + row_byte_width];
        for (out, bytes) in out_row.iter_mut().zip(src_row.as_chunks::<2>().0) {
            // Round, don't truncate: truncation shifts every sample down by half a step.
            *out = ((u16::from_le_bytes(*bytes) + half) >> shift).min(255) as u8;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use av1_grain::v_frame::{chroma::ChromaSubsampling, frame::FrameBuilder};

    use super::copy_decoded_plane_to_u8;

    #[test]
    fn ten_bit_samples_round_to_eight_bit() {
        let mut frame = FrameBuilder::new(4, 2, ChromaSubsampling::Monochrome, 8)
            .build::<u8>()
            .expect("valid frame");
        let samples: [u16; 8] = [0, 1, 2, 5, 6, 513, 1022, 1023];
        let bytes: Vec<u8> = samples.iter().flat_map(|v| v.to_le_bytes()).collect();
        copy_decoded_plane_to_u8(&bytes, 8, &mut frame.y_plane, 2).expect("copy");
        let out: Vec<u8> = frame.y_plane.pixels().collect();
        assert_eq!(out, [0, 0, 1, 1, 2, 128, 255, 255]);
    }
}

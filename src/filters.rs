use std::num::{NonZeroU8, NonZeroUsize};

use anyhow::{Result, anyhow, bail};
use av1_grain::v_frame::{
    chroma::ChromaSubsampling as Subsampling07,
    frame::{Frame, FrameBuilder as FrameBuilder07},
    pixel::Pixel,
    plane::Plane as Plane07,
};
use v_frame_05::{
    chroma::ChromaSubsampling as Subsampling05, frame::Frame as Frame05,
    frame::FrameBuilder as FrameBuilder05, pixel::Pixel as Pixel05, plane::Plane as Plane05,
};
use video_resize::algorithms::{
    BicubicCatmullRom, BicubicHermite, BicubicMitchell, Lanczos3, Spline36,
};
use video_resize::{CropDimensions, ResizeAlgorithm, ResizeDimensions, crop, resize};

pub struct FilterChain {
    filters: Vec<Filter>,
}

impl FilterChain {
    pub fn new(filters: &str) -> Result<Self> {
        if filters.is_empty() {
            return Ok(Self {
                filters: Vec::new(),
            });
        }

        let mut parsed = Vec::new();
        for filter in filters.split(';') {
            let (filter, args) = filter
                .split_once(':')
                .ok_or_else(|| anyhow!("Invalid filter syntax in \"{filter}\""))?;
            let args = args.split(',');
            match filter {
                "crop" => {
                    let (mut top, mut bottom, mut left, mut right) = (0, 0, 0, 0);
                    for arg in args {
                        let (arg, value) = arg
                            .split_once('=')
                            .ok_or_else(|| anyhow!("Invalid filter syntax in \"{arg}\""))?;
                        match arg {
                            "top" => {
                                top = value.parse()?;
                            }
                            "bottom" => {
                                bottom = value.parse()?;
                            }
                            "left" => {
                                left = value.parse()?;
                            }
                            "right" => {
                                right = value.parse()?;
                            }
                            arg => bail!("Unrecognized crop arg \"{arg}\""),
                        }
                    }
                    parsed.push(Filter::Crop {
                        top,
                        bottom,
                        left,
                        right,
                    });
                }
                "resize" => {
                    let (mut width, mut height, mut alg) = (0, 0, "catmullrom");
                    for arg in args {
                        let (arg, value) = arg
                            .split_once('=')
                            .ok_or_else(|| anyhow!("Invalid filter syntax in \"{arg}\""))?;
                        match arg {
                            "width" => {
                                width = value.parse()?;
                            }
                            "height" => {
                                height = value.parse()?;
                            }
                            "alg" => match value {
                                "hermite" => {
                                    alg = "hermite";
                                }
                                "catmullrom" => {
                                    alg = "catmullrom";
                                }
                                "mitchell" => {
                                    alg = "mitchell";
                                }
                                "lanczos" => {
                                    alg = "lanczos";
                                }
                                "spline36" => {
                                    alg = "spline36";
                                }
                                alg => bail!("Unrecognized resize algorithm \"{alg}\""),
                            },
                            arg => bail!("Unrecognized resize arg \"{arg}\""),
                        }
                    }
                    if width == 0 || height == 0 {
                        bail!("Both width and height must be provided to resize filter");
                    }
                    // SAFETY: checked above
                    unsafe {
                        parsed.push(Filter::Resize {
                            width: NonZeroUsize::new_unchecked(width),
                            height: NonZeroUsize::new_unchecked(height),
                            alg,
                        });
                    }
                }
                f => bail!("Unrecognized filter \"{f}\""),
            }
        }

        Ok(Self { filters: parsed })
    }

    pub fn apply<T: Pixel + Pixel05>(&self, frame: Frame<T>, source_bd: NonZeroU8) -> Frame<T> {
        self.filters
            .iter()
            .fold(frame, |prev, f| f.apply(&prev, source_bd))
    }

    #[must_use]
    pub fn is_crop_only(&self) -> bool {
        self.filters
            .iter()
            .all(|filter| matches!(filter, Filter::Crop { .. }))
    }
}

enum Filter {
    Crop {
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
    },
    Resize {
        width: NonZeroUsize,
        height: NonZeroUsize,
        alg: &'static str,
    },
}

impl Filter {
    pub fn apply<T: Pixel + Pixel05>(&self, frame: &Frame<T>, source_bd: NonZeroU8) -> Frame<T> {
        match *self {
            Filter::Crop {
                top,
                bottom,
                left,
                right,
            } => crop_frame(
                frame,
                CropDimensions {
                    top,
                    bottom,
                    left,
                    right,
                },
                source_bd,
            ),
            Filter::Resize { width, height, alg } => match alg {
                "hermite" => resize_frame::<T, BicubicHermite>(
                    frame,
                    ResizeDimensions { width, height },
                    source_bd,
                ),
                "catmullrom" => resize_frame::<T, BicubicCatmullRom>(
                    frame,
                    ResizeDimensions { width, height },
                    source_bd,
                ),
                "mitchell" => resize_frame::<T, BicubicMitchell>(
                    frame,
                    ResizeDimensions { width, height },
                    source_bd,
                ),
                "lanczos" => resize_frame::<T, Lanczos3>(
                    frame,
                    ResizeDimensions { width, height },
                    source_bd,
                ),
                "spline36" => resize_frame::<T, Spline36>(
                    frame,
                    ResizeDimensions { width, height },
                    source_bd,
                ),
                _ => unreachable!(),
            },
        }
    }
}

/// Runs `video_resize` (which speaks v_frame 0.5) on a v_frame 0.7 frame by
/// converting across the crate boundary in both directions. Pixel values are
/// preserved exactly; only the container types change.
fn crop_frame<T>(frame: &Frame<T>, dimensions: CropDimensions, source_bd: NonZeroU8) -> Frame<T>
where
    T: Pixel + Pixel05,
{
    let converted = frame_to_05(frame, source_bd.get()).unwrap();
    let cropped = crop(&converted, dimensions).unwrap();
    frame_from_05(&cropped, source_bd.get()).unwrap()
}

/// See [`crop_frame`]: resize runs on converted frames so output pixels match
/// `video-resize` output exactly.
fn resize_frame<T, F>(
    frame: &Frame<T>,
    dimensions: ResizeDimensions,
    source_bd: NonZeroU8,
) -> Frame<T>
where
    T: Pixel + Pixel05,
    F: ResizeAlgorithm,
{
    let converted = frame_to_05(frame, source_bd.get()).unwrap();
    let resized = resize::<T, F>(&converted, dimensions, source_bd).unwrap();
    frame_from_05(&resized, source_bd.get()).unwrap()
}

fn subsampling_to_05(subsampling: Subsampling07) -> Subsampling05 {
    match subsampling {
        Subsampling07::Yuv420 => Subsampling05::Yuv420,
        Subsampling07::Yuv422 => Subsampling05::Yuv422,
        Subsampling07::Yuv444 => Subsampling05::Yuv444,
        Subsampling07::Monochrome => Subsampling05::Monochrome,
    }
}

fn subsampling_from_05(subsampling: Subsampling05) -> Subsampling07 {
    match subsampling {
        Subsampling05::Yuv420 => Subsampling07::Yuv420,
        Subsampling05::Yuv422 => Subsampling07::Yuv422,
        Subsampling05::Yuv444 => Subsampling07::Yuv444,
        Subsampling05::Monochrome => Subsampling07::Monochrome,
    }
}

fn copy_plane_to_05<T>(source: &Plane07<T>, dest: &mut Plane05<T>) -> Result<()>
where
    T: Pixel + Pixel05,
{
    let converted: Vec<T> = source
        .rows()
        .flat_map(|row| row.iter())
        .map(|pixel| {
            let value: u16 = (*pixel).into();
            num_traits::NumCast::from(value).expect("identical bit depths convert losslessly")
        })
        .collect();
    dest.copy_from_slice(&converted)
        .map_err(|error| anyhow!("plane conversion failed: {error}"))?;
    Ok(())
}

fn copy_plane_from_05<T>(source: &Plane05<T>, dest: &mut Plane07<T>) -> Result<()>
where
    T: Pixel + Pixel05,
{
    let converted: Vec<T> = source
        .rows()
        .flat_map(|row| row.iter())
        .map(|pixel| {
            let value: u16 = num_traits::ToPrimitive::to_u16(pixel)
                .expect("identical bit depths convert losslessly");
            num_traits::NumCast::from(value).expect("identical bit depths convert losslessly")
        })
        .collect();
    dest.copy_from_slice(&converted)
        .map_err(|error| anyhow!("plane conversion failed: {error}"))?;
    Ok(())
}

fn frame_to_05<T>(frame: &Frame<T>, bit_depth: u8) -> Result<Frame05<T>>
where
    T: Pixel + Pixel05,
{
    let width = frame.y_plane.width();
    let height = frame.y_plane.height();
    let mut converted: Frame05<T> = FrameBuilder05::new(
        NonZeroUsize::new(width).ok_or_else(|| anyhow!("cannot convert empty frame"))?,
        NonZeroUsize::new(height).ok_or_else(|| anyhow!("cannot convert empty frame"))?,
        subsampling_to_05(frame.subsampling),
        NonZeroU8::new(bit_depth).ok_or_else(|| anyhow!("cannot convert zero-bit-depth frame"))?,
    )
    .build()
    .map_err(|error| anyhow!("frame conversion failed: {error}"))?;
    copy_plane_to_05(&frame.y_plane, &mut converted.y_plane)?;
    if let (Some(source), Some(dest)) = (frame.u_plane.as_ref(), converted.u_plane.as_mut()) {
        copy_plane_to_05(source, dest)?;
    }
    if let (Some(source), Some(dest)) = (frame.v_plane.as_ref(), converted.v_plane.as_mut()) {
        copy_plane_to_05(source, dest)?;
    }
    Ok(converted)
}

fn frame_from_05<T>(frame: &Frame05<T>, bit_depth: u8) -> Result<Frame<T>>
where
    T: Pixel + Pixel05,
{
    let width = frame.y_plane.width().get();
    let height = frame.y_plane.height().get();
    let mut converted: Frame<T> = FrameBuilder07::new(
        width,
        height,
        subsampling_from_05(frame.subsampling),
        bit_depth,
    )
    .build()
    .map_err(|error| anyhow!("frame conversion failed: {error}"))?;
    copy_plane_from_05(&frame.y_plane, &mut converted.y_plane)?;
    if let (Some(source), Some(dest)) = (frame.u_plane.as_ref(), converted.u_plane.as_mut()) {
        copy_plane_from_05(source, dest)?;
    }
    if let (Some(source), Some(dest)) = (frame.v_plane.as_ref(), converted.v_plane.as_mut()) {
        copy_plane_from_05(source, dest)?;
    }
    Ok(converted)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_new_error_contains(filters: &str, expected: &str) {
        let Err(err) = FilterChain::new(filters) else {
            panic!("expected parsing error for \"{filters}\"")
        };
        let message = err.to_string();
        assert!(
            message.contains(expected),
            "expected error to contain \"{expected}\", got \"{message}\""
        );
    }

    #[test]
    fn new_accepts_empty_filter_chain() {
        let chain = FilterChain::new("").unwrap();
        assert!(chain.filters.is_empty());
    }

    #[test]
    fn new_parses_crop_filter_args() {
        let chain = FilterChain::new("crop:top=1,bottom=2,left=3,right=4").unwrap();
        assert_eq!(chain.filters.len(), 1);

        match &chain.filters[0] {
            Filter::Crop {
                top,
                bottom,
                left,
                right,
            } => {
                assert_eq!(*top, 1);
                assert_eq!(*bottom, 2);
                assert_eq!(*left, 3);
                assert_eq!(*right, 4);
            }
            Filter::Resize { .. } => panic!("expected crop filter"),
        }
    }

    #[test]
    fn new_parses_resize_filter_with_default_algorithm() {
        let chain = FilterChain::new("resize:width=1920,height=1080").unwrap();
        assert_eq!(chain.filters.len(), 1);

        match &chain.filters[0] {
            Filter::Resize { width, height, alg } => {
                assert_eq!(width.get(), 1920);
                assert_eq!(height.get(), 1080);
                assert_eq!(*alg, "catmullrom");
            }
            Filter::Crop { .. } => panic!("expected resize filter"),
        }
    }

    #[test]
    fn new_parses_resize_filter_with_all_supported_algorithms() {
        for alg in ["hermite", "catmullrom", "mitchell", "lanczos", "spline36"] {
            let filter = format!("resize:width=640,height=360,alg={alg}");
            let chain = FilterChain::new(&filter).unwrap();

            match &chain.filters[0] {
                Filter::Resize {
                    width,
                    height,
                    alg: parsed_alg,
                } => {
                    assert_eq!(width.get(), 640);
                    assert_eq!(height.get(), 360);
                    assert_eq!(*parsed_alg, alg);
                }
                Filter::Crop { .. } => panic!("expected resize filter"),
            }
        }
    }

    #[test]
    fn new_parses_multiple_filters_in_order() {
        let chain = FilterChain::new("crop:top=4;resize:width=320,height=240,alg=lanczos").unwrap();
        assert_eq!(chain.filters.len(), 2);

        match &chain.filters[0] {
            Filter::Crop {
                top,
                bottom,
                left,
                right,
            } => {
                assert_eq!(*top, 4);
                assert_eq!(*bottom, 0);
                assert_eq!(*left, 0);
                assert_eq!(*right, 0);
            }
            Filter::Resize { .. } => panic!("expected crop filter"),
        }

        match &chain.filters[1] {
            Filter::Resize { width, height, alg } => {
                assert_eq!(width.get(), 320);
                assert_eq!(height.get(), 240);
                assert_eq!(*alg, "lanczos");
            }
            Filter::Crop { .. } => panic!("expected resize filter"),
        }
    }

    #[test]
    fn new_rejects_filter_without_colon_separator() {
        assert_new_error_contains("crop", "Invalid filter syntax in \"crop\"");
    }

    #[test]
    fn new_rejects_unrecognized_filter() {
        assert_new_error_contains("rotate:degrees=90", "Unrecognized filter \"rotate\"");
    }

    #[test]
    fn new_rejects_crop_arg_without_equals_separator() {
        assert_new_error_contains("crop:top", "Invalid filter syntax in \"top\"");
    }

    #[test]
    fn new_rejects_unrecognized_crop_arg() {
        assert_new_error_contains("crop:width=12", "Unrecognized crop arg \"width\"");
    }

    #[test]
    fn new_rejects_non_numeric_crop_value() {
        assert_new_error_contains("crop:top=abc", "invalid digit found in string");
    }

    #[test]
    fn new_rejects_resize_arg_without_equals_separator() {
        assert_new_error_contains(
            "resize:width=640,height",
            "Invalid filter syntax in \"height\"",
        );
    }

    #[test]
    fn new_rejects_unrecognized_resize_arg() {
        assert_new_error_contains(
            "resize:width=640,height=360,scale=2",
            "Unrecognized resize arg \"scale\"",
        );
    }

    #[test]
    fn new_rejects_unrecognized_resize_algorithm() {
        assert_new_error_contains(
            "resize:width=640,height=360,alg=nearest",
            "Unrecognized resize algorithm \"nearest\"",
        );
    }

    #[test]
    fn new_rejects_resize_when_width_or_height_missing() {
        assert_new_error_contains(
            "resize:width=640",
            "Both width and height must be provided to resize filter",
        );
        assert_new_error_contains(
            "resize:height=360",
            "Both width and height must be provided to resize filter",
        );
    }

    #[test]
    fn new_rejects_non_numeric_resize_dimensions() {
        assert_new_error_contains(
            "resize:width=wide,height=360",
            "invalid digit found in string",
        );
        assert_new_error_contains(
            "resize:width=640,height=tall",
            "invalid digit found in string",
        );
    }

    fn pattern(x: usize, y: usize, plane: usize) -> u16 {
        ((x * 3 + y * 5 + (x * y) % 17 + plane * 101) % 1024) as u16
    }

    fn build_frame_07<T>(width: usize, height: usize, bit_depth: u8) -> Frame<T>
    where
        T: Pixel + Pixel05,
    {
        use FrameBuilder07 as Builder;
        let mut frame: Frame<T> = Builder::new(width, height, Subsampling07::Yuv420, bit_depth)
            .build()
            .expect("valid frame");
        let max: u16 = if bit_depth == 8 { 255 } else { 1023 };
        for plane_index in 0..3 {
            let (w, h) = {
                let plane = frame.plane(plane_index).expect("plane exists");
                (plane.width(), plane.height())
            };
            let plane = frame.plane_mut(plane_index).expect("plane exists");
            for (y, row) in plane.rows_mut().enumerate().take(h) {
                for (x, pixel) in row.iter_mut().enumerate().take(w) {
                    let value = pattern(x, y, plane_index) % (max + 1);
                    *pixel = num_traits::NumCast::from(value).expect("in range");
                }
            }
        }
        frame
    }

    fn build_frame_05<T>(width: usize, height: usize, bit_depth: u8) -> Frame05<T>
    where
        T: Pixel05,
    {
        let mut frame: Frame05<T> = FrameBuilder05::new(
            NonZeroUsize::new(width).expect("nonzero"),
            NonZeroUsize::new(height).expect("nonzero"),
            Subsampling05::Yuv420,
            NonZeroU8::new(bit_depth).expect("nonzero"),
        )
        .build()
        .expect("valid frame");
        let max: u16 = if bit_depth == 8 { 255 } else { 1023 };
        for plane_index in 0..3 {
            let (w, h) = {
                let plane = frame.plane(plane_index).expect("plane exists");
                (plane.width().get(), plane.height().get())
            };
            let plane = frame.plane_mut(plane_index).expect("plane exists");
            for (y, row) in plane.rows_mut().enumerate().take(h) {
                for (x, pixel) in row.iter_mut().enumerate().take(w) {
                    let value = pattern(x, y, plane_index) % (max + 1);
                    *pixel = num_traits::NumCast::from(value).expect("in range");
                }
            }
        }
        frame
    }

    fn flat_pixels_07<T>(frame: &Frame<T>) -> Vec<u16>
    where
        T: Pixel,
    {
        let mut out = Vec::new();
        for plane_index in 0..3 {
            let plane = frame.plane(plane_index).expect("plane exists");
            let (w, h) = (plane.width(), plane.height());
            for row in plane.rows().take(h) {
                for pixel in row.iter().take(w) {
                    let value: u16 = (*pixel).into();
                    out.push(value);
                }
            }
        }
        out
    }

    fn flat_pixels_05<T>(frame: &Frame05<T>) -> Vec<u16>
    where
        T: Pixel05,
    {
        let mut out = Vec::new();
        for plane_index in 0..3 {
            let plane = frame.plane(plane_index).expect("plane exists");
            let (w, h) = (plane.width().get(), plane.height().get());
            for row in plane.rows().take(h) {
                for pixel in row.iter().take(w) {
                    out.push(num_traits::NumCast::from(*pixel).expect("u16 holds all"));
                }
            }
        }
        out
    }

    #[test]
    fn shim_crop_zero_round_trip_is_lossless() {
        for (width, height, bit_depth) in [(64usize, 64usize, 8u8), (62, 30, 10)] {
            let source_bd = NonZeroU8::new(bit_depth).expect("nonzero");
            let chain = FilterChain::new("crop:top=0,bottom=0,left=0,right=0").unwrap();
            if bit_depth == 8 {
                let frame = build_frame_07::<u8>(width, height, bit_depth);
                let expected = flat_pixels_07(&frame);
                let actual = flat_pixels_07(&chain.apply(frame, source_bd));
                assert_eq!(actual, expected, "u8 round-trip {width}x{height}");
            } else {
                let frame = build_frame_07::<u16>(width, height, bit_depth);
                let expected = flat_pixels_07(&frame);
                let actual = flat_pixels_07(&chain.apply(frame, source_bd));
                assert_eq!(actual, expected, "u16 round-trip {width}x{height}");
            }
        }
    }

    #[test]
    fn shim_resize_matches_direct_video_resize() {
        use video_resize::algorithms::Lanczos3;

        for (bit_depth, is_u8) in [(8u8, true), (10, false)] {
            let source_bd = NonZeroU8::new(bit_depth).expect("nonzero");
            let chain = FilterChain::new("resize:width=32,height=32,alg=lanczos").unwrap();
            let target = ResizeDimensions {
                width: NonZeroUsize::new(32).expect("nonzero"),
                height: NonZeroUsize::new(32).expect("nonzero"),
            };
            if is_u8 {
                let frame = build_frame_07::<u8>(64, 64, bit_depth);
                let shimmed = flat_pixels_07(&chain.apply(frame, source_bd));
                let oracle_source = build_frame_05::<u8>(64, 64, bit_depth);
                let oracle =
                    video_resize::resize::<u8, Lanczos3>(&oracle_source, target, source_bd)
                        .expect("oracle resizes");
                assert_eq!(shimmed, flat_pixels_05(&oracle), "u8 resize parity");
            } else {
                let frame = build_frame_07::<u16>(64, 64, bit_depth);
                let shimmed = flat_pixels_07(&chain.apply(frame, source_bd));
                let oracle_source = build_frame_05::<u16>(64, 64, bit_depth);
                let oracle =
                    video_resize::resize::<u16, Lanczos3>(&oracle_source, target, source_bd)
                        .expect("oracle resizes");
                assert_eq!(shimmed, flat_pixels_05(&oracle), "u16 resize parity");
            }
        }
    }
}

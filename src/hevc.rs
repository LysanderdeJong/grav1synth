use std::cmp::Ordering;

use anyhow::{Result, anyhow, bail};
use arrayvec::ArrayVec;
use av1_grain::{
    NUM_UV_COEFFS, NUM_UV_POINTS, NUM_Y_COEFFS, NUM_Y_POINTS, v_frame::chroma::ChromaSubsampling,
};
use ffmpeg::{Dictionary, Packet, Rational, codec, encoder, format::context::Output, media};
use log::debug;
use num_rational::Rational32;
use rayon::prelude::*;

use crate::{GrainTableSegment, parser::grain::FilmGrainParams, reader::BitstreamReader};

const NAL_PREFIX_SEI: u8 = 39;
const NAL_SUFFIX_SEI: u8 = 40;
const SEI_USER_DATA_REGISTERED_ITU_T_T35: u64 = 4;
const HEVC_PACKET_BATCH: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PacketFormat {
    AnnexB,
    LengthPrefixed { length_size: usize },
}

pub fn inspect_hevc_grain_table(input: &std::path::Path) -> Result<Vec<GrainTableSegment>> {
    let mut reader = BitstreamReader::open(input)?;
    ensure_hevc(&reader)?;
    let stream_idx = reader.get_video_stream()?.index();
    let frame_rate = reader.get_video_details().frame_rate;
    let video_stream_time_base = reader.input().stream(stream_idx as _).unwrap().time_base();
    let mut params_by_frame = Vec::new();
    let mut packet_data = Vec::with_capacity(HEVC_PACKET_BATCH);
    let mut packet_pts = Vec::with_capacity(HEVC_PACKET_BATCH);
    for (stream, packet) in reader.input().packets().filter_map(Result::ok) {
        if stream.index() != stream_idx {
            continue;
        }
        if let Some(data) = packet.data() {
            if packet_has_vcl(data)? {
                packet_data.push(data.to_vec());
                packet_pts.push(packet.pts().map(|pts| ffmpeg_pts_to_grain_ts(pts, video_stream_time_base)));
            }
        }
        if packet_data.len() >= HEVC_PACKET_BATCH {
            params_by_frame.extend(packet_pts.drain(..).zip(parse_hevc_grain_batch(&packet_data)?));
            packet_data.clear();
        }
    }
    params_by_frame.extend(packet_pts.drain(..).zip(parse_hevc_grain_batch(&packet_data)?));

    if params_by_frame.iter().any(|(pts, _)| pts.is_some()) {
        params_by_frame.sort_by_key(|(pts, _)| pts.unwrap_or(0));
    }
    let params_by_frame = params_by_frame
        .into_iter()
        .map(|(_, params)| params)
        .collect::<Vec<_>>();

    Ok(aggregate_hevc_grain(&params_by_frame, frame_rate))
}

pub fn has_hevc_grain(input: &std::path::Path) -> Result<bool> {
    let mut reader = BitstreamReader::open(input)?;
    ensure_hevc(&reader)?;
    let stream_idx = reader.get_video_stream()?.index();

    let mut packet_data = Vec::with_capacity(HEVC_PACKET_BATCH);
    for (stream, packet) in reader.input().packets().filter_map(Result::ok) {
        if stream.index() != stream_idx {
            continue;
        }
        if let Some(data) = packet.data() {
            packet_data.push(data.to_vec());
        }
        if packet_data.len() >= HEVC_PACKET_BATCH {
            if hevc_grain_batch_has_grain(&packet_data)? {
                return Ok(true);
            }
            packet_data.clear();
        }
    }

    hevc_grain_batch_has_grain(&packet_data)
}

pub fn modify_hevc_grain(
    input: &std::path::Path,
    output: &std::path::Path,
    segments: Option<&[GrainTableSegment]>,
    replace: bool,
) -> Result<()> {
    let mut reader = BitstreamReader::open(input)?;
    ensure_hevc(&reader)?;

    let width = reader.get_video_details().width;
    let height = reader.get_video_details().height;
    let chroma_sampling = reader.get_video_details().chroma_sampling;
    let stream_idx = reader.get_video_stream()?.index();
    let ictx = reader.input();
    let mut writer = ffmpeg::format::output(output)?;
    let mut stream_mapping = vec![0; ictx.nb_streams() as _];
    let mut ist_time_bases = vec![Rational(0, 1); ictx.nb_streams() as _];
    let mut ost_index = 0;
    let presentation_timestamps = if segments.is_some() {
        presentation_timestamps_by_decode_index(input)?
    } else {
        Vec::new()
    };

    let input_chapters: Vec<(i64, Rational, i64, i64, Dictionary)> = ictx
        .chapters()
        .map(|ch| {
            (
                ch.id(),
                ch.time_base(),
                ch.start(),
                ch.end(),
                ch.metadata().to_owned(),
            )
        })
        .collect();

    for (ist_index, ist) in ictx.streams().enumerate() {
        let ist_medium = ist.parameters().medium();
        if ist_medium != media::Type::Audio
            && ist_medium != media::Type::Video
            && ist_medium != media::Type::Subtitle
        {
            stream_mapping[ist_index] = -1;
            continue;
        }
        stream_mapping[ist_index] = ost_index;
        ist_time_bases[ist_index] = ist.time_base();
        ost_index += 1isize;

        let ist_metadata = ist.metadata().to_owned();
        let ist_sar = ist.sample_aspect_ratio();

        let mut ost = writer.add_stream(encoder::find(codec::Id::None)).unwrap();
        ost.set_parameters(ist.parameters());
        ost.metadata_mut().replace_with(ist_metadata);
        ost.set_sample_aspect_ratio(ist_sar);
        unsafe {
            let ist_stream = ist.as_ptr();
            let ost_stream = ost.as_mut_ptr();
            let ost_params = ost.parameters_mut().as_mut_ptr();
            (*ost_params).codec_tag = 0;
            (*ost_stream).r_frame_rate = (*ist_stream).r_frame_rate;
            (*ost_stream).time_base = (*ist_stream).time_base;
            (*ost_stream).disposition = (*ist_stream).disposition;
        }
    }

    writer
        .metadata_mut()
        .replace_with(ictx.metadata().to_owned());
    for (id, time_base, start, end, metadata) in input_chapters {
        let title = metadata.as_ref().get("title").unwrap_or("").to_owned();
        let mut out_chapter = writer.add_chapter(id, time_base, start, end, &title)?;
        out_chapter.metadata_mut().replace_with(metadata);
    }

    let video_stream_time_base = ictx.stream(stream_idx as _).unwrap().time_base();
    writer.write_header()?;

    let mut packet_entries = Vec::with_capacity(HEVC_PACKET_BATCH);
    let mut video_jobs = Vec::with_capacity(HEVC_PACKET_BATCH);
    let mut video_packet_index = 0usize;
    for (stream, packet) in ictx.packets().filter_map(Result::ok) {
        let video_job_index = if stream.index() == stream_idx {
            packet.data().map(|data| {
                let params = if packet_has_vcl(data).unwrap_or(false) {
                    let packet_ts = presentation_timestamps
                        .get(video_packet_index)
                        .copied()
                        .flatten()
                        .unwrap_or_else(|| {
                            ffmpeg_pts_to_grain_ts(
                                packet.pts().unwrap_or_default(),
                                video_stream_time_base,
                            )
                        });
                    video_packet_index += 1;
                    segments
                        .and_then(|segments| segment_for_ts(segments, packet_ts))
                        .cloned()
                } else {
                    None
                };
                let index = video_jobs.len();
                video_jobs.push(VideoPacketJob {
                    data: data.to_vec(),
                    params,
                    modified: Vec::new(),
                });
                index
            })
        } else {
            None
        };
        packet_entries.push(PacketEntry {
            stream_index: stream.index(),
            packet,
            video_job_index,
        });
        if packet_entries.len() >= HEVC_PACKET_BATCH {
            process_and_write_packet_batch(
                &mut packet_entries,
                &mut video_jobs,
                &mut writer,
                &stream_mapping,
                &ist_time_bases,
                replace,
                width,
                height,
                chroma_sampling,
            )?;
        }
    }
    process_and_write_packet_batch(
        &mut packet_entries,
        &mut video_jobs,
        &mut writer,
        &stream_mapping,
        &ist_time_bases,
        replace,
        width,
        height,
        chroma_sampling,
    )?;

    writer.write_trailer()?;
    Ok(())
}

struct PacketEntry {
    stream_index: usize,
    packet: Packet,
    video_job_index: Option<usize>,
}

struct VideoPacketJob {
    data: Vec<u8>,
    params: Option<FilmGrainParams>,
    modified: Vec<u8>,
}

fn parse_hevc_grain_batch(packet_data: &[Vec<u8>]) -> Result<Vec<Option<FilmGrainParams>>> {
    packet_data
        .par_iter()
        .map(|data| parse_packet_afgs1_params(data))
        .collect()
}

fn hevc_grain_batch_has_grain(packet_data: &[Vec<u8>]) -> Result<bool> {
    Ok(packet_data
        .par_iter()
        .map(|data| Ok(find_nals(data)?.iter().any(|nal| is_afgs1_sei(nal.data))))
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .any(|found| found))
}

#[allow(clippy::too_many_arguments)]
fn process_and_write_packet_batch(
    packet_entries: &mut Vec<PacketEntry>,
    video_jobs: &mut Vec<VideoPacketJob>,
    writer: &mut Output,
    stream_mapping: &[isize],
    ist_time_bases: &[Rational],
    replace: bool,
    width: usize,
    height: usize,
    chroma_sampling: ChromaSubsampling,
) -> Result<()> {
    video_jobs.par_iter_mut().try_for_each(|job| {
        job.modified = modify_packet(
            &job.data,
            job.params.as_ref(),
            replace,
            width,
            height,
            chroma_sampling,
        )?;
        Ok::<_, anyhow::Error>(())
    })?;

    for entry in packet_entries.drain(..) {
        let stream_index = entry.stream_index;
        let mut packet = entry.packet;
        if let Some(video_job_index) = entry.video_job_index {
            let modified = &video_jobs[video_job_index].modified;
            let pts = packet.pts();
            let dts = packet.dts();
            let duration = packet.duration();
            match modified.len().cmp(&packet.size()) {
                Ordering::Greater => packet.grow(modified.len() - packet.size()),
                Ordering::Less => packet.shrink(modified.len()),
                Ordering::Equal => (),
            }
            packet.set_pts(pts);
            packet.set_dts(dts);
            packet.set_duration(duration);
            packet.data_mut().unwrap().copy_from_slice(modified);
        }

        write_packet(writer, packet, stream_index, stream_mapping, ist_time_bases)?;
    }
    video_jobs.clear();

    Ok(())
}

fn ensure_hevc(reader: &BitstreamReader) -> Result<()> {
    let codec_id = reader.get_video_stream()?.parameters().id();
    if codec_id != codec::Id::HEVC {
        bail!("HEVC/x265 grain SEI editing requires an HEVC video stream; found {codec_id:?}");
    }
    Ok(())
}

fn ffmpeg_pts_to_grain_ts(pts: i64, time_base: Rational) -> u64 {
    if pts < 0 {
        return 0;
    }
    let pts = pts as u64;
    let num = time_base.0 as u64;
    let den = time_base.1 as u64;
    if den == 0 {
        return 0;
    }
    (pts * num * 10_000_000u64).div_ceil(den)
}

fn frame_index_to_grain_ts(frame_index: usize, frame_rate: Rational32) -> u64 {
    let numer = i64::from(*frame_rate.numer());
    let denom = i64::from(*frame_rate.denom());
    if numer <= 0 || denom <= 0 {
        return 0;
    }
    (((frame_index as u128) * (denom as u128) * 10_000_000u128) / (numer as u128)) as u64
}

fn presentation_timestamps_by_decode_index(input: &std::path::Path) -> Result<Vec<Option<u64>>> {
    let mut reader = BitstreamReader::open(input)?;
    ensure_hevc(&reader)?;
    let stream_idx = reader.get_video_stream()?.index();
    let frame_rate = reader.get_video_details().frame_rate;

    let mut packets = Vec::new();
    for (stream, packet) in reader.input().packets().filter_map(Result::ok) {
        if stream.index() == stream_idx
            && let Some(data) = packet.data()
        {
            if packet_has_vcl(data)? {
                packets.push((packets.len(), packet.pts()));
            }
        }
    }
    if packets.iter().any(|(_, pts)| pts.is_none()) {
        return Ok(vec![None; packets.len()]);
    }

    packets.sort_by(|(left_index, left_pts), (right_index, right_pts)| {
        left_pts
            .cmp(right_pts)
            .then_with(|| left_index.cmp(right_index))
    });

    let mut timestamps = vec![None; packets.len()];
    for (frame_index, (decode_index, _)) in packets.into_iter().enumerate() {
        timestamps[decode_index] = Some(frame_index_to_grain_ts(frame_index, frame_rate));
    }

    Ok(timestamps)
}

fn segment_for_ts(segments: &[GrainTableSegment], timestamp: u64) -> Option<&FilmGrainParams> {
    segments
        .iter()
        .find(|segment| segment.start_time <= timestamp && timestamp < segment.end_time)
        .map(|segment| &segment.grain_params)
}

fn write_packet(
    writer: &mut Output,
    mut packet: Packet,
    ist_index: usize,
    stream_mapping: &[isize],
    ist_time_bases: &[Rational],
) -> Result<()> {
    let ost_index = stream_mapping[ist_index];
    if ost_index < 0 {
        return Ok(());
    }
    let ost = writer.stream(ost_index as _).unwrap();
    packet.rescale_ts(ist_time_bases[ist_index], ost.time_base());
    packet.set_position(-1);
    packet.set_stream(ost_index as _);
    packet.write_interleaved(writer)?;
    Ok(())
}

fn modify_packet(
    data: &[u8],
    params: Option<&FilmGrainParams>,
    replace: bool,
    width: usize,
    height: usize,
    chroma_sampling: ChromaSubsampling,
) -> Result<Vec<u8>> {
    let nals = find_nals(data)?;
    if nals.is_empty() {
        return Ok(data.to_vec());
    }

    let has_existing = nals.iter().any(|nal| is_afgs1_sei(nal.data));
    if has_existing && params.is_some() && !replace {
        bail!("HEVC stream already contains AFGS1 SEI; re-run with --replace to replace it");
    }

    let mut out = Vec::with_capacity(data.len() + 256);
    let mut inserted = false;
    let format = nals[0].format;
    let sei = params
        .map(|params| build_afgs1_nal(params, format, width, height, chroma_sampling))
        .transpose()?;

    for nal in nals {
        let nal_type = nal_type(nal.data).ok();
        if !inserted && sei.is_some() && nal_type.is_some_and(is_vcl_nal) {
            out.extend_from_slice(sei.as_ref().unwrap());
            inserted = true;
        }
        if is_afgs1_sei(nal.data) {
            debug!("Removing existing HEVC AFGS1 SEI");
            continue;
        }
        out.extend_from_slice(nal.prefix);
        out.extend_from_slice(nal.data);
    }

    Ok(out)
}

#[derive(Debug, Clone, Copy)]
struct Nal<'a> {
    prefix: &'a [u8],
    data: &'a [u8],
    format: PacketFormat,
}

fn find_nals(data: &[u8]) -> Result<Vec<Nal<'_>>> {
    if let Ok(nals) = find_length_prefixed_nals(data, 4) {
        return Ok(nals);
    }
    if starts_with_annex_b(data) {
        Ok(find_annex_b_nals(data))
    } else {
        find_length_prefixed_nals(data, 4)
    }
}

fn starts_with_annex_b(data: &[u8]) -> bool {
    data.starts_with(&[0, 0, 1]) || data.starts_with(&[0, 0, 0, 1])
}

fn find_annex_b_nals(data: &[u8]) -> Vec<Nal<'_>> {
    let mut starts = Vec::new();
    let mut i = 0usize;
    while i + 3 <= data.len() {
        if data[i..].starts_with(&[0, 0, 1]) {
            starts.push((i, 3));
            i += 3;
        } else if i + 4 <= data.len() && data[i..].starts_with(&[0, 0, 0, 1]) {
            starts.push((i, 4));
            i += 4;
        } else {
            i += 1;
        }
    }

    let mut nals = Vec::with_capacity(starts.len());
    for (idx, (start, prefix_len)) in starts.iter().copied().enumerate() {
        let end = starts.get(idx + 1).map_or(data.len(), |(next, _)| *next);
        nals.push(Nal {
            prefix: &data[start..start + prefix_len],
            data: &data[start + prefix_len..end],
            format: PacketFormat::AnnexB,
        });
    }
    nals
}

fn find_length_prefixed_nals(data: &[u8], length_size: usize) -> Result<Vec<Nal<'_>>> {
    let mut nals = Vec::new();
    let mut pos = 0usize;
    while pos < data.len() {
        if pos + length_size > data.len() {
            bail!("malformed HEVC packet: incomplete NAL length");
        }
        let len = data[pos..pos + length_size]
            .iter()
            .fold(0usize, |acc, byte| (acc << 8) | usize::from(*byte));
        let nal_start = pos + length_size;
        let nal_end = nal_start + len;
        if len == 0 || nal_end > data.len() {
            bail!("malformed HEVC packet: invalid NAL length {len}");
        }
        nals.push(Nal {
            prefix: &data[pos..nal_start],
            data: &data[nal_start..nal_end],
            format: PacketFormat::LengthPrefixed { length_size },
        });
        pos = nal_end;
    }
    Ok(nals)
}

fn nal_type(nal: &[u8]) -> Result<u8> {
    if nal.len() < 2 {
        bail!("malformed HEVC NAL: missing header");
    }
    Ok((nal[0] >> 1) & 0x3f)
}

fn is_vcl_nal(nal_type: u8) -> bool {
    nal_type <= 31
}

fn packet_has_vcl(data: &[u8]) -> Result<bool> {
    Ok(find_nals(data)?
        .iter()
        .any(|nal| nal_type(nal.data).is_ok_and(is_vcl_nal)))
}

fn is_afgs1_sei(nal: &[u8]) -> bool {
    matches!(nal_type(nal), Ok(NAL_PREFIX_SEI | NAL_SUFFIX_SEI))
        && parse_sei_messages(nal).is_ok_and(|messages| {
            messages.iter().any(|(payload_type, payload)| {
                *payload_type == SEI_USER_DATA_REGISTERED_ITU_T_T35 && is_afgs1_payload(payload)
            })
        })
}

fn parse_sei_messages(nal: &[u8]) -> Result<Vec<(u64, Vec<u8>)>> {
    let rbsp = ebsp_to_rbsp(
        nal.get(2..)
            .ok_or_else(|| anyhow!("missing HEVC NAL header"))?,
    );
    let mut pos = 0usize;
    let mut messages = Vec::new();
    while pos + 1 < rbsp.len() {
        if rbsp[pos] == 0x80 && rbsp[pos + 1..].iter().all(|b| *b == 0) {
            break;
        }
        let mut payload_type = 0u64;
        while pos < rbsp.len() && rbsp[pos] == 0xff {
            payload_type += 255;
            pos += 1;
        }
        if pos >= rbsp.len() {
            break;
        }
        payload_type += u64::from(rbsp[pos]);
        pos += 1;

        let mut payload_size = 0usize;
        while pos < rbsp.len() && rbsp[pos] == 0xff {
            payload_size += 255;
            pos += 1;
        }
        if pos >= rbsp.len() {
            break;
        }
        payload_size += usize::from(rbsp[pos]);
        pos += 1;
        if pos + payload_size > rbsp.len() {
            break;
        }
        messages.push((payload_type, rbsp[pos..pos + payload_size].to_vec()));
        pos += payload_size;
    }
    Ok(messages)
}

fn is_afgs1_payload(payload: &[u8]) -> bool {
    payload.len() >= 5
        && payload[0] == 0xb5
        && payload[1] == 0x58
        && payload[2] == 0x90
        && payload[3] == 0x01
        && (payload[4] & 0x80) != 0
}

fn parse_packet_afgs1_params(data: &[u8]) -> Result<Option<FilmGrainParams>> {
    for nal in find_nals(data)? {
        if !matches!(nal_type(nal.data), Ok(NAL_PREFIX_SEI | NAL_SUFFIX_SEI)) {
            continue;
        }
        for (payload_type, payload) in parse_sei_messages(nal.data)? {
            if payload_type == SEI_USER_DATA_REGISTERED_ITU_T_T35 && is_afgs1_payload(&payload) {
                return parse_afgs1_payload(&payload).map(Some);
            }
        }
    }
    Ok(None)
}

fn parse_afgs1_payload(payload: &[u8]) -> Result<FilmGrainParams> {
    let mut bits = BitReader::new(payload);
    let country_code = bits.read(8)?;
    let provider_code = bits.read(16)?;
    let provider_oriented_code = bits.read(8)?;
    if country_code != 0xb5 || provider_code != 0x5890 || provider_oriented_code != 0x01 {
        bail!("not an x265 AFGS1 payload");
    }
    if bits.read(1)? == 0 {
        bail!("AFGS1 payload is disabled");
    }
    bits.read(4)?; // reserved_4bits
    let sets_minus1 = bits.read(3)?;
    if sets_minus1 != 0 {
        bail!("only one AFGS1 film grain set is supported");
    }
    let payload_less_than_4byte = bits.read(1)? != 0;
    let payload_bits = if payload_less_than_4byte { 2 } else { 8 };
    bits.read(payload_bits)?; // payload_size
    bits.read(3)?; // film_grain_param_set_idx
    let apply_grain = bits.read(1)? != 0;
    if !apply_grain {
        bail!("AFGS1 payload does not apply grain");
    }
    let grain_seed = bits.read(16)? as u16;
    let update_grain = bits.read(1)? != 0;
    if !update_grain {
        bail!("AFGS1 reference/copy grain records are not supported");
    }
    bits.read(4)?; // apply_units_resolution_log2
    bits.read(12)?; // apply_horz_resolution
    bits.read(12)?; // apply_vert_resolution
    let luma_only = bits.read(1)? != 0;
    if !luma_only {
        bits.read(1)?; // subsampling_x
        bits.read(1)?; // subsampling_y
    }
    let video_signal_characteristics = bits.read(1)? != 0;
    if video_signal_characteristics {
        bits.read(3)?; // bit_depth_minus8
        let cicp = bits.read(1)? != 0;
        if cicp {
            bits.read(8)?; // colour_primaries
            bits.read(8)?; // transfer_characteristics
            bits.read(8)?; // matrix_coefficients
            bits.read(1)?; // video_full_range_flag
        }
    }
    let predict_scaling = bits.read(1)? != 0;
    if predict_scaling {
        bail!("AFGS1 predict_scaling_flag is not supported by grav1synth inspect yet");
    }

    let scaling_points_y = read_scaling_points::<NUM_Y_POINTS>(&mut bits, false)?;
    let chroma_scaling_from_luma = if luma_only { false } else { bits.read(1)? != 0 };
    let (scaling_points_cb, scaling_points_cr) = if luma_only || chroma_scaling_from_luma {
        (ArrayVec::new(), ArrayVec::new())
    } else {
        (
            read_scaling_points::<NUM_UV_POINTS>(&mut bits, true)?,
            read_scaling_points::<NUM_UV_POINTS>(&mut bits, true)?,
        )
    };
    let scaling_shift = bits.read(2)? as u8 + 8;
    let ar_coeff_lag = bits.read(2)? as u8;
    let ar_coeffs_y = if scaling_points_y.is_empty() {
        ArrayVec::new()
    } else {
        read_coeffs::<NUM_Y_COEFFS>(&mut bits, 24)?
    };
    let ar_coeffs_cb = if !scaling_points_cb.is_empty() || chroma_scaling_from_luma {
        read_coeffs::<NUM_UV_COEFFS>(&mut bits, 25)?
    } else {
        ArrayVec::new()
    };
    let ar_coeffs_cr = if !scaling_points_cr.is_empty() || chroma_scaling_from_luma {
        read_coeffs::<NUM_UV_COEFFS>(&mut bits, 25)?
    } else {
        ArrayVec::new()
    };
    let ar_coeff_shift = bits.read(2)? as u8 + 6;
    let grain_scale_shift = bits.read(2)? as u8;
    let (cb_mult, cb_luma_mult, cb_offset) = if !scaling_points_cb.is_empty() {
        (
            bits.read(8)? as u8,
            bits.read(8)? as u8,
            bits.read(9)? as u16,
        )
    } else {
        (0, 0, 0)
    };
    let (cr_mult, cr_luma_mult, cr_offset) = if !scaling_points_cr.is_empty() {
        (
            bits.read(8)? as u8,
            bits.read(8)? as u8,
            bits.read(9)? as u16,
        )
    } else {
        (0, 0, 0)
    };
    let overlap_flag = bits.read(1)? != 0;
    let clip_to_restricted_range = bits.read(1)? != 0;

    Ok(FilmGrainParams {
        grain_seed,
        scaling_points_y,
        scaling_points_cb,
        scaling_points_cr,
        scaling_shift,
        ar_coeff_lag,
        ar_coeffs_y,
        ar_coeffs_cb,
        ar_coeffs_cr,
        ar_coeff_shift,
        cb_mult,
        cb_luma_mult,
        cb_offset,
        cr_mult,
        cr_luma_mult,
        cr_offset,
        chroma_scaling_from_luma,
        grain_scale_shift,
        overlap_flag,
        clip_to_restricted_range,
    })
}

fn read_scaling_points<const N: usize>(
    bits: &mut BitReader<'_>,
    chroma: bool,
) -> Result<ArrayVec<[u8; 2], N>> {
    let count = bits.read(4)? as usize;
    let mut points = ArrayVec::new();
    if count == 0 {
        return Ok(points);
    }
    let value_bits = bits.read(3)? as usize + 1;
    let scaling_bits = bits.read(2)? as usize + 5;
    if chroma {
        bits.read(8)?; // cb/cr_scaling_offset
    }
    let mut prev = 0u8;
    for idx in 0..count {
        let increment = bits.read(value_bits)? as u8;
        let value = if idx == 0 {
            increment
        } else {
            prev.wrapping_add(increment)
        };
        let scaling = bits.read(scaling_bits)? as u8;
        points.push([value, scaling]);
        prev = value;
    }
    Ok(points)
}

fn read_coeffs<const N: usize>(bits: &mut BitReader<'_>, count: usize) -> Result<ArrayVec<i8, N>> {
    let coeff_bits = bits.read(2)? as usize + 5;
    let mut coeffs = ArrayVec::new();
    for _ in 0..count {
        let encoded = bits.read(coeff_bits)? as i16;
        coeffs.push((encoded - (1 << (coeff_bits - 1))) as i8);
    }
    Ok(coeffs)
}

fn aggregate_hevc_grain(
    params_by_frame: &[Option<FilmGrainParams>],
    frame_rate: Rational32,
) -> Vec<GrainTableSegment> {
    let mut segments: Vec<GrainTableSegment> = Vec::new();

    for (frame_index, params) in params_by_frame.iter().enumerate() {
        let cur_packet_start = frame_index_to_grain_ts(frame_index, frame_rate);
        let cur_packet_end = frame_index_to_grain_ts(frame_index + 1, frame_rate);
        if let Some(params) = params {
            if let Some(cur_segment) = segments.last_mut()
                && cur_segment.end_time == cur_packet_start
                && params == &cur_segment.grain_params
            {
                cur_segment.end_time = cur_packet_end;
            } else {
                segments.push(GrainTableSegment {
                    start_time: cur_packet_start,
                    end_time: cur_packet_end,
                    grain_params: params.clone(),
                });
            }
        }
    }

    if let Some(last) = segments.last_mut() {
        last.end_time = i64::MAX as u64;
    }

    segments
}

fn build_afgs1_nal(
    params: &FilmGrainParams,
    format: PacketFormat,
    width: usize,
    height: usize,
    chroma_sampling: ChromaSubsampling,
) -> Result<Vec<u8>> {
    if params.chroma_scaling_from_luma {
        bail!(
            "HEVC AFGS1 injection does not support chroma_scaling_from_luma; use explicit Cb/Cr scaling points"
        );
    }
    let payload = build_afgs1_payload(params, width, height, chroma_sampling)?;
    let mut rbsp = Vec::new();
    write_sei_message_header(SEI_USER_DATA_REGISTERED_ITU_T_T35, payload.len(), &mut rbsp)?;
    rbsp.extend_from_slice(&payload);
    rbsp.push(0x80);

    let mut nal = vec![NAL_PREFIX_SEI << 1, 1];
    nal.extend_from_slice(&rbsp_to_ebsp(&rbsp));

    Ok(match format {
        PacketFormat::AnnexB => {
            let mut out = vec![0, 0, 0, 1];
            out.extend_from_slice(&nal);
            out
        }
        PacketFormat::LengthPrefixed { length_size } => {
            let mut out = Vec::with_capacity(length_size + nal.len());
            write_length_prefix(nal.len(), length_size, &mut out)?;
            out.extend_from_slice(&nal);
            out
        }
    })
}

fn write_sei_message_header(
    payload_type: u64,
    payload_size: usize,
    out: &mut Vec<u8>,
) -> Result<()> {
    let mut payload_type = payload_type;
    while payload_type >= 255 {
        out.push(255);
        payload_type -= 255;
    }
    out.push(payload_type.try_into()?);

    let mut payload_size = payload_size;
    while payload_size >= 255 {
        out.push(255);
        payload_size -= 255;
    }
    out.push(payload_size.try_into()?);
    Ok(())
}

fn build_afgs1_payload(
    params: &FilmGrainParams,
    width: usize,
    height: usize,
    chroma_sampling: ChromaSubsampling,
) -> Result<Vec<u8>> {
    let (units_resolution_log2, horz_resolution, vert_resolution) =
        film_grain_resolution(width, height)?;
    let mut payload_size = 0usize;
    let mut payload_bits = 2usize;
    let mut payload = Vec::new();

    for _ in 0..4 {
        payload.clear();
        let mut bits = BitWriter::new(&mut payload);
        bits.write(0xb5, 8)?;
        bits.write(0x5890, 16)?;
        bits.write(0x01, 8)?;
        bits.write(1, 1)?; // afgs1_enable_flag
        bits.write(0, 4)?; // reserved_4bits
        bits.write(0, 3)?; // num_film_grain_sets_minus1
        bits.write(u64::from(payload_size < 4), 1)?;
        bits.write(payload_size as u64, payload_bits)?;
        write_afgs1_params(
            &mut bits,
            params,
            units_resolution_log2,
            horz_resolution,
            vert_resolution,
            chroma_sampling,
        )?;
        bits.byte_align()?;

        let new_size = payload.len().saturating_sub(4);
        let new_bits = if new_size < 4 { 2 } else { 8 };
        if new_size == payload_size && new_bits == payload_bits {
            return Ok(payload);
        }
        payload_size = new_size;
        payload_bits = new_bits;
    }

    Ok(payload)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn find_nals_prefers_valid_length_prefix_over_annex_b_lookalike() {
        let mut packet = vec![0x00, 0x00, 0x01, 0xe6, 0x02, 0x01];
        packet.resize(490, 0);

        let nals = find_nals(&packet).expect("length-prefixed packet should parse");

        assert_eq!(nals.len(), 1);
        assert_eq!(nals[0].format, PacketFormat::LengthPrefixed { length_size: 4 });
        assert_eq!(nal_type(nals[0].data).unwrap(), 1);
    }

    #[test]
    fn frame_index_timestamps_match_x265_aom_floor_mapping() {
        let frame_rate = Rational32::new(24_000, 1_001);

        assert_eq!(frame_index_to_grain_ts(27, frame_rate), 11_261_250);
        assert_eq!(frame_index_to_grain_ts(28, frame_rate), 11_678_333);
    }
}

fn write_afgs1_params(
    bits: &mut BitWriter<'_, '_>,
    params: &FilmGrainParams,
    units_resolution_log2: usize,
    horz_resolution: usize,
    vert_resolution: usize,
    chroma_sampling: ChromaSubsampling,
) -> Result<()> {
    let luma_only = params.scaling_points_cb.is_empty() && params.scaling_points_cr.is_empty();
    bits.write(0, 3)?; // film_grain_param_set_idx
    bits.write(1, 1)?; // apply_grain_flag
    bits.write(u64::from(params.grain_seed), 16)?;
    bits.write(1, 1)?; // update_grain_flag
    bits.write(units_resolution_log2 as u64, 4)?;
    bits.write(horz_resolution as u64, 12)?;
    bits.write(vert_resolution as u64, 12)?;
    bits.write(u64::from(luma_only), 1)?;
    if !luma_only {
        let (subsampling_x, subsampling_y) = match chroma_sampling {
            ChromaSubsampling::Yuv420 => (true, true),
            ChromaSubsampling::Yuv422 => (true, false),
            ChromaSubsampling::Yuv444 => (false, false),
            ChromaSubsampling::Monochrome => {
                bail!("chroma grain cannot be injected into monochrome HEVC")
            }
        };
        bits.write(u64::from(subsampling_x), 1)?;
        bits.write(u64::from(subsampling_y), 1)?;
    }
    bits.write(0, 1)?; // video_signal_characteristics_flag
    bits.write(0, 1)?; // predict_scaling_flag

    write_scaling_points(bits, &params.scaling_points_y, false)?;
    if !luma_only {
        bits.write(0, 1)?; // chroma_scaling_from_luma
        write_scaling_points(bits, &params.scaling_points_cb, true)?;
        write_scaling_points(bits, &params.scaling_points_cr, true)?;
    }

    bits.write(u64::from(params.scaling_shift.saturating_sub(8)), 2)?;
    bits.write(u64::from(params.ar_coeff_lag), 2)?;
    if !params.scaling_points_y.is_empty() {
        write_coeffs(bits, params.ar_coeffs_y.iter().copied(), 24)?;
    }
    if !params.scaling_points_cb.is_empty() {
        write_coeffs(bits, params.ar_coeffs_cb.iter().copied(), 25)?;
    }
    if !params.scaling_points_cr.is_empty() {
        write_coeffs(bits, params.ar_coeffs_cr.iter().copied(), 25)?;
    }
    bits.write(u64::from(params.ar_coeff_shift.saturating_sub(6)), 2)?;
    bits.write(u64::from(params.grain_scale_shift), 2)?;
    if !params.scaling_points_cb.is_empty() {
        bits.write(params.cb_mult as u8 as u64, 8)?;
        bits.write(params.cb_luma_mult as u8 as u64, 8)?;
        bits.write((i32::from(params.cb_offset) & 0x1ff) as u64, 9)?;
    }
    if !params.scaling_points_cr.is_empty() {
        bits.write(params.cr_mult as u8 as u64, 8)?;
        bits.write(params.cr_luma_mult as u8 as u64, 8)?;
        bits.write((i32::from(params.cr_offset) & 0x1ff) as u64, 9)?;
    }
    bits.write(u64::from(params.overlap_flag), 1)?;
    bits.write(u64::from(params.clip_to_restricted_range), 1)?;
    Ok(())
}

fn write_scaling_points(
    bits: &mut BitWriter<'_, '_>,
    points: &[[u8; 2]],
    chroma: bool,
) -> Result<()> {
    bits.write(points.len().try_into()?, 4)?;
    if points.is_empty() {
        return Ok(());
    }
    bits.write(7, 3)?; // point_value_increment_bits_minus1, x265 hard-codes 8 bits
    bits.write(3, 2)?; // point_scaling_bits_minus5, x265 hard-codes 8 bits
    if chroma {
        bits.write(0, 8)?; // cb/cr_scaling_offset
    }
    let mut prev = 0u8;
    for (idx, point) in points.iter().enumerate() {
        let value = if idx == 0 {
            point[0]
        } else {
            point[0].wrapping_sub(prev)
        };
        bits.write(u64::from(value), 8)?;
        bits.write(u64::from(point[1]), 8)?;
        prev = point[0];
    }
    Ok(())
}

fn write_coeffs(
    bits: &mut BitWriter<'_, '_>,
    coeffs: impl IntoIterator<Item = i8>,
    count: usize,
) -> Result<()> {
    bits.write(3, 2)?; // bits_per_ar_coeff_minus5, x265 hard-codes 8 bits
    let mut written = 0usize;
    for coeff in coeffs.into_iter().take(count) {
        bits.write((i32::from(coeff) + 128) as u64, 8)?;
        written += 1;
    }
    for _ in written..count {
        bits.write(128, 8)?;
    }
    Ok(())
}

fn film_grain_resolution(width: usize, height: usize) -> Result<(usize, usize, usize)> {
    if width == 0 || height == 0 {
        bail!("zero-sized HEVC stream is unsupported");
    }
    let log2 = width.trailing_zeros().min(height.trailing_zeros()) as usize;
    let unit = 1usize << log2;
    Ok((log2, width / unit, height / unit))
}

struct BitWriter<'a, 'b> {
    out: &'a mut Vec<u8>,
    cur: u8,
    used: u8,
    _marker: std::marker::PhantomData<&'b ()>,
}

struct BitReader<'a> {
    data: &'a [u8],
    bit_pos: usize,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, bit_pos: 0 }
    }

    fn read(&mut self, bits: usize) -> Result<u64> {
        if bits > 64 || self.bit_pos + bits > self.data.len() * 8 {
            bail!("AFGS1 payload ended early while reading {bits} bits");
        }
        let mut value = 0u64;
        for _ in 0..bits {
            let byte = self.data[self.bit_pos / 8];
            let shift = 7 - (self.bit_pos % 8);
            value = (value << 1) | u64::from((byte >> shift) & 1);
            self.bit_pos += 1;
        }
        Ok(value)
    }
}

impl<'a, 'b> BitWriter<'a, 'b> {
    fn new(out: &'a mut Vec<u8>) -> Self {
        Self {
            out,
            cur: 0,
            used: 0,
            _marker: std::marker::PhantomData,
        }
    }

    fn write(&mut self, value: u64, bits: usize) -> Result<()> {
        if bits < 64 && value >= (1u64 << bits) {
            bail!("value {value} does not fit in {bits} bits");
        }
        for shift in (0..bits).rev() {
            let bit = ((value >> shift) & 1) as u8;
            self.cur = (self.cur << 1) | bit;
            self.used += 1;
            if self.used == 8 {
                self.out.push(self.cur);
                self.cur = 0;
                self.used = 0;
            }
        }
        Ok(())
    }

    fn byte_align(&mut self) -> Result<()> {
        if self.used != 0 {
            self.write(1, 1)?;
            while self.used != 0 {
                self.write(0, 1)?;
            }
        }
        Ok(())
    }
}

fn ebsp_to_rbsp(ebsp: &[u8]) -> Vec<u8> {
    let mut rbsp = Vec::with_capacity(ebsp.len());
    let mut zeros = 0u8;
    for byte in ebsp {
        if zeros >= 2 && *byte == 0x03 {
            zeros = 0;
            continue;
        }
        rbsp.push(*byte);
        zeros = if *byte == 0 { zeros + 1 } else { 0 };
    }
    rbsp
}

fn rbsp_to_ebsp(rbsp: &[u8]) -> Vec<u8> {
    let mut ebsp = Vec::with_capacity(rbsp.len());
    let mut zeros = 0u8;
    for byte in rbsp {
        if zeros >= 2 && *byte <= 0x03 {
            ebsp.push(0x03);
            zeros = 0;
        }
        ebsp.push(*byte);
        zeros = if *byte == 0 { zeros + 1 } else { 0 };
    }
    ebsp
}

fn write_length_prefix(len: usize, length_size: usize, out: &mut Vec<u8>) -> Result<()> {
    if length_size == 0 || length_size > 4 || len >= (1usize << (length_size * 8)) {
        bail!("invalid HEVC NAL length prefix size {length_size}");
    }
    for shift in (0..length_size).rev() {
        out.push(((len >> (shift * 8)) & 0xff) as u8);
    }
    Ok(())
}

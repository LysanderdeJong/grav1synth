#!/usr/bin/env bash
# Generates tiny, byte-reproducible test fixtures for grav1synth.
#
# Usage: bash tests/fixtures/make.sh [outdir]
#   outdir defaults to <script dir>/media. A relative outdir is taken
#   relative to the current working directory.
#
# Requires ffmpeg/ffprobe with libsvtav1, libx265, ffv1, libopus and aac.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT_DIR=${1:-"$SCRIPT_DIR/media"}
mkdir -p "$OUT_DIR"
OUT_DIR=$(cd "$OUT_DIR" && pwd)

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

SIZE=128x72
# Video is cut by input duration, not -frames:v, so the 1 s audio is not truncated.
# 20 frames at 24 fps (or 23.976 fps) fit in 0.82 s.
VDUR=0.82
FF=(ffmpeg -hide_banner -loglevel error -y)
# Bit-exact muxing/encoding: no encoder/app strings, no random UIDs.
BITEXACT=(-fflags +bitexact -flags:v +bitexact -flags:a +bitexact)
# Seeded noise (default seed is random, which would break reruns).
NOISE="noise=alls=12:allf=t+u:all_seed=1"
# Variable frame rate: frames 0-9 are 1000 ticks apart, the rest 1500 (tick = 1/90000 s).
VFR="settb=1/90000,setpts='if(lt(N,10),N*1000,10000+(N-10)*1500)'"
TB90K=(-video_track_timescale 90000 -fps_mode passthrough)

SRT="$WORK/subs.srt"
cat > "$SRT" <<'EOF'
1
00:00:00,000 --> 00:00:00,400
First subtitle cue

2
00:00:00,500 --> 00:00:00,800
Second subtitle cue
EOF

META="$WORK/chapters.ffmeta"
cat > "$META" <<'EOF'
;FFMETADATA1
title=grav1synth fixture
[CHAPTER]
TIMEBASE=1/1000
START=0
END=400
title=Chapter one
[CHAPTER]
TIMEBASE=1/1000
START=400
END=800
title=Chapter two
EOF

# Inputs per fixture are ordered: 0 = video, 1 = audio, 2 = SRT, 3 = chapters
# (chapter/metadata input is 2 when there are no subtitles).
lavfi_video() { # rate
  printf 'testsrc2=size=%s:rate=%s' "$SIZE" "$1"
}
SINE="sine=frequency=440:sample_rate=48000:duration=1"

echo "Writing fixtures to $OUT_DIR"

# 1. SVT-AV1 8-bit, 24 fps CFR, audio + subtitles + chapters.
"${FF[@]}" \
  -t "$VDUR" -f lavfi -i "$(lavfi_video 24)" \
  -f lavfi -i "$SINE" \
  -f srt -i "$SRT" \
  -f ffmetadata -i "$META" \
  -map 0:v:0 -map 1:a:0 -map 2:s:0 -map_metadata 3 -map_chapters 3 \
  -vf "$NOISE,format=yuv420p" -fps_mode cfr \
  -c:v libsvtav1 -preset 8 -crf 35 -pix_fmt yuv420p -svtav1-params lp=1 -threads 1 \
  -c:a libopus -b:a 64k -c:s srt \
  "${BITEXACT[@]}" "$OUT_DIR/av1_cfr8.mkv"

# 2. SVT-AV1 10-bit, VFR, audio + chapters (MP4, 90 kHz track timescale).
"${FF[@]}" \
  -t "$VDUR" -f lavfi -i "$(lavfi_video 24)" \
  -f lavfi -i "$SINE" \
  -f ffmetadata -i "$META" \
  -map 0:v:0 -map 1:a:0 -map_metadata 2 -map_chapters 2 \
  -vf "$NOISE,format=yuv420p10le,$VFR" \
  -c:v libsvtav1 -preset 8 -crf 35 -pix_fmt yuv420p10le -enc_time_base 1/90000 -svtav1-params lp=1 -threads 1 \
  -c:a aac -b:a 64k "${TB90K[@]}" \
  "${BITEXACT[@]}" "$OUT_DIR/av1_vfr10.mp4"

# 3. SVT-AV1 8-bit, 24000/1001 fps, non-zero start time, audio.
"${FF[@]}" \
  -t "$VDUR" -f lavfi -i "$(lavfi_video 24000/1001)" \
  -f lavfi -i "$SINE" \
  -map 0:v:0 -map 1:a:0 \
  -vf "$NOISE,format=yuv420p" -fps_mode cfr \
  -c:v libsvtav1 -preset 8 -crf 35 -pix_fmt yuv420p -svtav1-params lp=1 -threads 1 \
  -c:a libopus -b:a 64k -output_ts_offset 10 \
  "${BITEXACT[@]}" "$OUT_DIR/av1_start.mkv"

# 4. SVT-AV1 8-bit with encoder-side film grain (grain already in the stream).
"${FF[@]}" \
  -t "$VDUR" -f lavfi -i "$(lavfi_video 24)" \
  -map 0:v:0 \
  -vf "$NOISE,format=yuv420p" -fps_mode cfr \
  -c:v libsvtav1 -preset 6 -crf 35 -pix_fmt yuv420p -svtav1-params film-grain=8:lp=1 -threads 1 \
  "${BITEXACT[@]}" "$OUT_DIR/av1_grain.mkv"

# 5. x265 8-bit, B-frames, 24 fps CFR, audio + subtitles + chapters.
"${FF[@]}" \
  -t "$VDUR" -f lavfi -i "$(lavfi_video 24)" \
  -f lavfi -i "$SINE" \
  -f srt -i "$SRT" \
  -f ffmetadata -i "$META" \
  -map 0:v:0 -map 1:a:0 -map 2:s:0 -map_metadata 3 -map_chapters 3 \
  -vf "$NOISE,format=yuv420p" -fps_mode cfr \
  -c:v libx265 -preset medium -crf 28 -pix_fmt yuv420p \
  -x265-params "bframes=3:pools=1:frame-threads=1:log-level=error" -threads 1 \
  -c:a libopus -b:a 64k -c:s srt \
  "${BITEXACT[@]}" "$OUT_DIR/hevc_cfr8.mkv"

# 6. x265 10-bit, B-frames, VFR, audio, hvc1 tag, 90 kHz track timescale (MP4).
"${FF[@]}" \
  -t "$VDUR" -f lavfi -i "$(lavfi_video 24)" \
  -f lavfi -i "$SINE" \
  -map 0:v:0 -map 1:a:0 \
  -vf "$NOISE,format=yuv420p10le,$VFR" \
  -c:v libx265 -preset medium -crf 28 -pix_fmt yuv420p10le -enc_time_base 1/90000 -tag:v hvc1 \
  -x265-params "bframes=3:pools=1:frame-threads=1:log-level=error" -threads 1 \
  -c:a aac -b:a 64k "${TB90K[@]}" \
  "${BITEXACT[@]}" "$OUT_DIR/hevc_vfr10.mp4"

# 7. x265 8-bit, B-frames, 24000/1001 fps, non-zero start time.
"${FF[@]}" \
  -t "$VDUR" -f lavfi -i "$(lavfi_video 24000/1001)" \
  -map 0:v:0 \
  -vf "$NOISE,format=yuv420p" -fps_mode cfr \
  -c:v libx265 -preset medium -crf 28 -pix_fmt yuv420p \
  -x265-params "bframes=3:pools=1:frame-threads=1:log-level=error" -threads 1 \
  -output_ts_offset 10 \
  "${BITEXACT[@]}" "$OUT_DIR/hevc_start.mp4"

# Denoised companions for grain analysis: decode, hqdn3d=6, FFV1, same pix_fmt, video only.
denoise() { # input-stem
  "${FF[@]}" \
    -i "$OUT_DIR/$1" \
    -map 0:v:0 -an -sn -dn \
    -vf "hqdn3d=6" -fps_mode passthrough \
    -c:v ffv1 -threads 1 \
    -map_metadata -1 -map_chapters -1 "${BITEXACT[@]}" "$OUT_DIR/${1%.*}.denoised.mkv"
}
denoise av1_cfr8.mkv
denoise av1_vfr10.mp4
denoise hevc_cfr8.mkv

echo "Done: $(ls "$OUT_DIR" | wc -l) files in $OUT_DIR"

#!/usr/bin/env bash
#
# Golden parity recorder for the grav1synth CLI.
#
#   Record:  bash tests/fixtures/record.sh <grav1synth-exe> <outdir> [mediadir]
#   Compare: bash tests/fixtures/record.sh --compare <dirA> <dirB>
#
# mediadir defaults to tests/fixtures/media next to this script.
#
# Each case is written to <outdir>/<stem>__<op>/ containing:
#   exit.txt        exit code of the CLI
#   stdout.txt      normalised stdout
#   stderr.txt      normalised stderr
#   produced.txt    files the case left in its work dir, or "(none)"
#   table.tbl       copy of out.tbl, only when a table was produced
#   probe.json      ffprobe stream/chapter description, only when media was produced
#   packets.json    ffprobe per-packet listing with md5 payload hashes
#   extradata.txt   hex dump lines of the first video stream's extradata
# Output media is deleted after probing. apply_table output is kept until
# inspect_after_apply has read it, then deleted with the other scratch files.
# Also written: <outdir>/_env.txt (exe md5, ffprobe path/version, input md5s).
# It is not part of the comparison.
#
# Normalisation (only things that cannot legitimately differ between runs):
#   stdout/stderr:
#     - ANSI CSI escape sequences removed.
#     - CR converted to LF and blank lines dropped (ffmpeg on Windows writes CRLF).
#     - progress/timing lines dropped (elapsed, ETA, it/s, fps=, speed=, time=,
#       "N ms", took, done in). Safety net: the binary prints none on a non-TTY.
#     - absolute paths replaced by tokens <WORK>, <OUT>, <MEDIA>, <REPO>. Each is
#       matched in unix (/d/x), mixed (D:/x) and Windows (D:\x) spellings, with
#       the drive letter in both cases. Reason: the output dir, the scratch dir
#       and the repo location differ between recordings.
#     - libav object pointers "[matroska @ 0000021493AC1E00]" and bare
#       0x... addresses replaced by <addr>. Reason: they change every run.
#   probe.json:
#     - values (not keys) of tags encoder/ENCODER, creation_time, DURATION*,
#       NUMBER_OF_FRAMES*, NUMBER_OF_BYTES*, BPS*, _STATISTICS_* replaced by
#       "<normalised>". Keys are kept, so a tag appearing or disappearing is
#       still caught. Reason: encoder/muxer version strings, wall-clock time,
#       and bitrates derived from file size.
#   probe.json, packets.json, extradata.txt: CR stripped (ffprobe writes CRLF on
#     Windows when stdout is redirected). Nothing else changed.
#   table.tbl: byte-exact copy of the CLI output.
#
# ffprobe is resolved from PATH before the ffmpeg-dev DLL directory is prepended
# for the CLI, so the probe tool itself is the same as before that change.

set -uo pipefail
shopt -s nullglob
export LC_ALL=C

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TABLE="$REPO_ROOT/tests/example-table.tbl"
FFDEV_BIN=/c/bin/ffmpeg-dev/ffmpeg-n9.0-latest-win64-gpl-shared-9.0/bin

CASES=0
FAILS=0
TOK_FROM=()
TOK_TO=()

die() { echo "record.sh: $*" >&2; exit 2; }

usage() {
    cat >&2 <<'EOF'
usage: bash tests/fixtures/record.sh <grav1synth-exe> <outdir> [mediadir]
       bash tests/fixtures/record.sh --compare <dirA> <dirB>
EOF
}

# ---------------------------------------------------------------- compare mode

list_case_dirs() {
    (cd "$1" && ls -1p | sed -n 's:/$::p')
}

compare() {
    local a="$1" b="$2" names c out pass=0 fail=0
    [ -d "$a" ] || die "not a directory: $a"
    [ -d "$b" ] || die "not a directory: $b"
    names="$( { list_case_dirs "$a"; list_case_dirs "$b"; } | sort -u )"
    [ -n "$names" ] || { echo "no case directories in $a or $b" >&2; return 1; }
    while IFS= read -r c; do
        if [ ! -d "$a/$c" ]; then
            echo "FAIL $c (missing in A)"
            fail=$((fail + 1))
        elif [ ! -d "$b/$c" ]; then
            echo "FAIL $c (missing in B)"
            fail=$((fail + 1))
        elif out="$(diff -r "$a/$c" "$b/$c" 2>&1)"; then
            echo "PASS $c"
            pass=$((pass + 1))
        else
            echo "FAIL $c"
            printf '%s\n' "$out" | head -n 20 | sed 's/^/     /'
            fail=$((fail + 1))
        fi
    done <<< "$names"
    echo "$pass passed, $fail failed"
    [ "$fail" -eq 0 ]
}

# ---------------------------------------------------------------- record mode

# Register every spelling a path can take in CLI output under one token.
add_tokens() { # <dir> <token>
    local unix win back p
    unix="$(cd "$1" && pwd)"
    win="$(cd "$1" && pwd -W)"
    back="${win//\//\\}"
    for p in "$unix" "$win" "$back"; do
        [ -n "$p" ] || continue
        TOK_FROM+=("$p"); TOK_TO+=("$2")
        TOK_FROM+=("$(printf '%s' "$p" | sed 's/^./\l&/')"); TOK_TO+=("$2")
    done
}

# Stdin -> stdout: normalise a log stream as described in the header.
normalise_log() {
    local s i
    s="$(cat; printf x)"; s="${s%x}"
    for i in "${!TOK_FROM[@]}"; do
        s="${s//"${TOK_FROM[$i]}"/"${TOK_TO[$i]}"}"
    done
    printf '%s' "$s" | tr '\r' '\n' | sed -E \
        -e 's/\x1b\[[0-9;?]*[ -/]*[@-~]//g' \
        -e 's/\[([A-Za-z0-9_]+) @ [0-9A-Fa-fx]+\]/[\1 @ <addr>]/g' \
        -e 's/0x[0-9A-Fa-f]{6,}/<addr>/g' \
        -e '/elapsed|\beta[: ]|it\/s|\bfps=|speed=|time=[0-9]|\b[0-9.]+ ?ms\b|\btook\b|done in/Id' \
        -e '/^$/d'
}

# Stdin -> stdout: blank the values of tags that legitimately vary.
normalise_tags() {
    sed -E 's/"(encoder|creation_time|duration(-[a-z]+)?|number_of_frames(-[a-z]+)?|number_of_bytes(-[a-z]+)?|bps(-[a-z]+)?|_statistics_[a-z_]+(-[a-z]+)?)"[[:space:]]*:[[:space:]]*"[^"]*"/"\1": "<normalised>"/Ig'
}

# ffprobe writes CRLF to a redirected stdout on Windows; strip it so the files are LF.
lf() { tr -d '\r'; }

probe_media() { # <file> <casedir>
    local f="$1" d="$2"
    "$PROBE" -v error -show_entries 'stream=index,codec_name,codec_type,codec_tag_string,profile,width,height,pix_fmt,sample_aspect_ratio,color_range,color_space,color_transfer,color_primaries,r_frame_rate,avg_frame_rate,time_base,start_pts,extradata_size:stream_disposition:stream_tags:format_tags:chapter=id,time_base,start,end:chapter_tags' \
        -show_chapters -of json "$f" 2>/dev/null | normalise_tags | lf > "$d/probe.json"
    "$PROBE" -v error -show_packets -show_data_hash md5 \
        -show_entries packet=stream_index,pts,dts,duration,flags,size,data_hash \
        -of json "$f" 2>/dev/null | lf > "$d/packets.json"
    "$PROBE" -v error -show_streams -show_data -select_streams v:0 "$f" 2>/dev/null \
        | sed -n '/^extradata=/,/^$/p' | grep -E '^[0-9a-fA-F]{8}: ' | lf > "$d/extradata.txt" || true
}

# record_case <case> <ext> <cli args...>
#   ext is "tbl" for table outputs, otherwise the media extension of the input.
#   The CLI runs with cwd = the case work dir and writes out.<ext>.
record_case() {
    local case="$1" ext="$2" code produced
    shift 2
    local d="$OUTDIR/$case" w="$WORK/$case"
    rm -rf "$d" "$w"
    mkdir -p "$d" "$w"

    ( cd "$w" && "$EXE" "$@" > "$LOGS/$case.stdout" 2> "$LOGS/$case.stderr" )
    code=$?

    printf '%s\n' "$code" > "$d/exit.txt"
    normalise_log < "$LOGS/$case.stdout" > "$d/stdout.txt"
    normalise_log < "$LOGS/$case.stderr" > "$d/stderr.txt"

    produced="$(cd "$w" && ls -A | sort)"
    if [ -n "$produced" ]; then
        printf '%s\n' "$produced" > "$d/produced.txt"
    else
        echo "(none)" > "$d/produced.txt"
    fi

    if [ "$ext" = tbl ]; then
        [ -f "$w/out.tbl" ] && cp "$w/out.tbl" "$d/table.tbl"
    elif [ -f "$w/out.$ext" ]; then
        probe_media "$w/out.$ext" "$d"
    fi

    # apply_table output is kept for inspect_after_apply; everything else goes now.
    case "$case" in
        *__apply_table) ;;
        *) rm -rf "$w" ;;
    esac

    CASES=$((CASES + 1))
    [ "$code" -eq 0 ] || FAILS=$((FAILS + 1))
    echo "  $case: exit=$code produced=$(printf '%s' "$produced" | tr '\n' ' ')"
}

record() {
    [ $# -ge 2 ] || { usage; exit 2; }
    [ -f "$1" ] || die "grav1synth exe not found: $1"
    [ -f "$TABLE" ] || die "missing $TABLE"

    EXE="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
    local media_in="${3:-$SCRIPT_DIR/media}"
    [ -d "$media_in" ] || die "media dir not found: $media_in"
    MEDIA="$(cd "$media_in" && pwd)"
    mkdir -p "$2"
    OUTDIR="$(cd "$2" && pwd)"

    WORK="$REPO_ROOT/target/golden-scratch/record-$$"
    LOGS="$WORK/logs"
    mkdir -p "$LOGS"
    trap 'rm -rf "$WORK"' EXIT

    # Resolve ffprobe before the DLL directory is prepended, so it stays the PATH one.
    PROBE="$(command -v ffprobe)" || die "ffprobe not on PATH"
    if [ -d "$FFDEV_BIN" ]; then
        export PATH="$FFDEV_BIN:$PATH"
    fi

    local probe_ver
    probe_ver="$("$PROBE" -version 2>/dev/null | head -n 1)"
    case "$probe_ver" in
        *"version n9.0.2"*|*"version 9.0.2"*) ;;
        *) echo "warning: ffprobe is not 9.0.2: $probe_ver" >&2 ;;
    esac
    {
        echo "exe: $EXE"
        echo "exe_md5: $(md5sum < "$EXE" | cut -d' ' -f1)"
        echo "ffprobe: $PROBE"
        echo "ffprobe_version: $probe_ver"
        echo "inputs:"
        (cd "$MEDIA" && md5sum -- * 2>/dev/null) | sed 's/^/  /'
    } > "$OUTDIR/_env.txt"

    add_tokens "$WORK" "<WORK>"
    add_tokens "$OUTDIR" "<OUT>"
    add_tokens "$MEDIA" "<MEDIA>"
    add_tokens "$REPO_ROOT" "<REPO>"

    local path base stem ext apply_out den dens s found p
    for path in "$MEDIA"/*; do
        [ -f "$path" ] || continue
        base="${path##*/}"
        case "$base" in *.denoised.*) continue ;; esac
        stem="${base%.*}"
        ext="$(printf '%s' "${base##*.}" | tr 'A-Z' 'a-z')"
        echo "== $base"

        record_case "${stem}__inspect" tbl inspect "$path" -o out.tbl -y
        record_case "${stem}__apply_table" "$ext" apply "$path" -o "out.$ext" -y --grain "$TABLE"
        record_case "${stem}__apply_preset" "$ext" apply "$path" -o "out.$ext" -y --preset 16mm
        record_case "${stem}__apply_iso" "$ext" apply "$path" -o "out.$ext" -y --iso 400 --chroma
        record_case "${stem}__apply_replace" "$ext" apply "$path" -o "out.$ext" -y --grain "$TABLE" --replace
        record_case "${stem}__remove" "$ext" remove "$path" -o "out.$ext" -y

        apply_out="$WORK/${stem}__apply_table/out.$ext"
        if [ -f "$apply_out" ]; then
            record_case "${stem}__inspect_after_apply" tbl inspect "$apply_out" -o out.tbl -y
        else
            echo "  ${stem}__inspect_after_apply: skipped (apply_table produced no output)"
        fi
        rm -rf "$WORK/${stem}__apply_table"

        dens=( "$MEDIA/${stem}".denoised.* )
        if [ "${#dens[@]}" -gt 1 ]; then
            echo "warning: several denoised companions for $stem, using ${dens[0]##*/}" >&2
        fi
        if [ "${#dens[@]}" -gt 0 ]; then
            den="${dens[0]}"
            record_case "${stem}__diff" tbl diff "$path" "$den" -o out.tbl -y
            record_case "${stem}__diff_crop" tbl diff "$path" "$den" -o out.tbl -y --filters "crop:top=8,left=8"
            record_case "${stem}__diff_resize" tbl diff "$path" "$den" -o out.tbl -y --filters "resize:width=64,height=36"
        fi
    done

    # Warn about denoised companions whose source is not in the media dir.
    for den in "$MEDIA"/*.denoised.*; do
        base="${den##*/}"
        s="${base%%.denoised.*}"
        found=0
        for p in "$MEDIA/$s".*; do
            case "$p" in *.denoised.*) ;; *) found=1 ;; esac
        done
        [ "$found" -eq 1 ] || echo "warning: no source for companion $base" >&2
    done

    echo "recorded $CASES cases into $OUTDIR ($FAILS with non-zero exit) in ${SECONDS}s"
}

# ---------------------------------------------------------------- entry point

if [ "${1:-}" = "--compare" ]; then
    [ $# -eq 3 ] || { usage; exit 2; }
    compare "$2" "$3"
    exit $?
fi

record "$@"

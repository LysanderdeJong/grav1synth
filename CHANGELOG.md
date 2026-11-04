## Unreleased

- Drop the FFmpeg libraries (`ffmpeg-the-third`): grav1synth now runs the `ffmpeg`/`ffprobe`
  executables instead. Building needs no FFmpeg headers or libclang. Video packets go to
  ffmpeg as NUT, so timestamps, B-frame dts, extradata and the other streams, chapters and
  metadata come out as before. The library API takes a `tools::Launcher`.
- `apply` and `remove` now update the container's AV1 configuration record (MP4 `av1C`,
  Matroska `CodecPrivate`) when they turn grain on or off, so its `film_grain_params_present`
  matches the bitstream.
- `diff` no longer drops frames when the AV1 decoder holds frames back, and fails on decode
  and read errors instead of ignoring them.
- `apply` maps AV1 packets to grain table segments by frame time as libaom does, undoing container rounding: in a 1 ms time base (Matroska) a frame stored as 83 ms is matched as 83.33 ms, so it no longer lands in the segment before. VFR timestamps are used as they are.
- `tests/fixtures`: fixture generator, golden recorder and goldens for parity checks.
- Support ffmpeg 9.0
- Merge `apply` and `generate` into a single `apply` command. `--grain <FILE>` applies
  table-based grain; `--iso <NUM>` (with optional `--chroma`) applies photon-noise-based grain.
- `apply` now checks for existing grain headers before writing. If grain is already present it
  prints a notice and skips, unless `--replace` is also provided. This makes it safe to use in
  automated workflows where existing grain should be preserved.

## Version 0.2.0

- Upgrade all of the internals to the latest rust-av crates
- Remove ISO range limit
- Support ffmpeg 8.0
- Extend packet to include adjustment obu size
- Fix writing `apply_grain` to false
- Add parsing for `global_param`

## Version 0.1.0-beta.6

- Add a progress bar for the `diff` function.
- Fix bug where some codecs may not return the final frame.
- Add ability to crop and resize `diff` sources.
- Fix issue where `diff` may produce bad grain tables due to decoder frame desyncs.

## Version 0.1.0-beta.5

- Fix issue where `diff` may fail in certain circumstances.
- Considerably speed up the `diff` command.

## Version 0.1.0-beta.4

- Fix issue where `apply` and `remove` commands did not modify the file correctly.

## Version 0.1.0-beta.3

- Fix compatibility with a number of videos.
  - There are still a few known files with issues, but a significantly larger number should work now.

## Version 0.1.0-beta.2

- Implement the "diff" command. See README for usage.

## Version 0.1.0-beta.1

- Implement the "apply" and "generate" commands. See README for usage.

## Version 0.1.0-alpha.4

- Implement the "remove" command, which removes all grain synthesis from a video

## Version 0.1.0-alpha.3

- Fix parsing bugs affecting some videos

## Version 0.1.0-alpha.2

- Fix the timestamp output to be in 1/10,000,000 of a second units

## Version 0.1.0-alpha.1

- Initial release
- Enables the `inspect` command

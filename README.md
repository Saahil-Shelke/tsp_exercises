# DIGIT Tactile Sensor – Data Exploration & Illumination Study

Python tools for working with recordings from a [DIGIT](https://digit.ml) vision-based tactile
sensor (serial `D20758`) recorded with the OpenTouch dashboard.

1. **Load** `.touch` recordings, **replay** the frames, and **save interesting frames** as PNG.
2. **Collect** new data while changing the **LED illumination**, **visualize** it, and
   answer *how do the images change?*

## Quick start

```bash
make setup          # uv venv + install requirements
make all            # info + interesting frames + illumination analysis + mp4 export
make help           # list every command
```

Manual use (without make):

```bash
source .venv/bin/activate
python save_interesting.py
python analyze_illumination.py
```

## Commands

| Command | What it does |
|---|---|
| `make setup` | Create `.venv` with **uv** and install `requirements.txt` |
| `make info` | Frames, duration, fps and LED config of every `.touch` file |
| `make replay FILE=... SPEED=...` | Replay in an OpenCV window (`space` pause, `a`/`d` step, `s` save PNG, `q` quit) |
| `make export FILE=...` | Write the replay as `outputs/replay/<name>.mp4` (no display needed) |
| `make export-all` | Export every recording as mp4 |
| `make frames TOP=3` | Save the strongest-contact frames per object → `outputs/frames/` |
| `make illumination` | Compare `recording_test.touch` against the contactless `recording.touch` → `outputs/illumination/` |
| `make collect SERIAL=D20758` | Live capture: global LED sweep 0…15, frames with and without contact at each level |
| `make collect-rgb` | Live capture: one LED colour at a time |
| `make collect-free` | Live capture: sweep without the press-an-object step |
| `make collect-sim` | Simulated sweep (no sensor), only for testing the pipeline |
| `make analyze-sweep [SWEEP=file.h5]` | Plot the latest (or a given) captured sweep |
| `make clean` / `make clean-all` | Delete outputs / also delete sweeps and the venv |

## Project layout

```
datasets/                  .touch recordings (HDF5)
  recording.touch          contactless reference (61 s)
  recording_band.touch     band   (12 s)
  recording_cent.touch     cent coin (20 s)
  recording_chip.touch     chip   (22 s)
  recording_euro.touch     euro coin (14 s, LED rgb=[8,15,15])
  recording_key.touch      key    (22 s)
  recording_test.touch     illumination changed in the dashboard (8.5 s)
touch_loader.py            load_touch(path) -> frames (N,320,240,3 BGR), timestamps
replay.py                  frame replay / mp4 export
save_interesting.py        automatic contact-frame selection
collect_digit.py           live capture with an LED intensity sweep (digit-interface)
analyze_illumination.py    illumination visualizations (recordings or sweep)
outputs/
  frames/                  <obj>_fXXXX.png, *_panel.png, <obj>_score.png, overview.png
  illumination/            brightness_timeline, compare_frames, histograms, ratio_maps, ...
  replay/                  mp4 exports
  collected/               illum_sweep_*.h5 from collect_digit.py
```

## The `.touch` file format

These files are HDF5 with three datasets:

| Dataset | Contents |
|---|---|
| `config` | JSON: sensor name, type, serial, LED `rgb` (0–15 each), resolution |
| `chunk_start_times` / `chunk_end_times` | about 5 s per chunk |
| `sensor_chunks` | variable-length `uint8` blob per chunk |

Each blob is little-endian:

```
u32 n_sensors, u32 len, sensor_name      ("Finger")
u32 n_streams, u32 len, stream_name      ("camera")
u32 n_frames
n_frames × {
    u32  record_len
    char name[32]
    f64  timestamp      (s since start)
    u32  rows=320, cols=240, channels=3
    u8   pixels[rows*cols*3]   (BGR)
}
```

Frames are QVGA at about 29.6 fps. A few frames per file are corrupted single-frame glitches.
`save_interesting.py` median-filters the contact score so these glitches are never picked.

## How it works

**Interesting frames.** The background is the median of the first 30 frames of each file, taken
before anything touches the gel. Each frame's contact score is the 99th percentile of
`|frame − background|`. The score is median-filtered over 7 frames, and the top peaks at least
1 s apart are saved. Each saved frame gets a panel: *no contact | contact | difference heat map*.

**Illumination analysis.**
- Mean brightness of each colour channel over time, plotted with the contact score so that
  lighting changes and object presses aren't confused.
- No-contact test frame compared against the reference: difference image, histograms, and
  per-channel ratio maps showing *where* the light changed.
- For a captured sweep: one image per level, mean brightness, % saturated pixels, temporal noise
  and contact contrast vs LED intensity.

## Results: how do the images change?

From `recording_test.touch` against `recording.touch`:

- **One clean illumination step at about 1.0 s**, with nothing touching the gel: blue +4.5,
  red +1.5, green unchanged. Before and after look almost identical to the eye.
- The test recording is brighter overall than the reference (B +4, G +7.6, R +4.6). The
  increase is **not uniform**: each colour brightens on the side where its LED sits, because the
  DIGIT lights the gel from the edges with separate R, G and B LEDs.
- Contact contrast is about the same at both settings.
- **The changes are much smaller than LED settings from 0 to 15 should cause.** The saved
  config still says `[15,15,15]`, and `recording_euro.touch` (`rgb=[8,15,15]`) shows no darker
  red channel. Either the dashboard change didn't fully apply, or the camera's auto-exposure is
  cancelling it out. A real `make collect` sweep will tell which.

Expected behaviour of a full sweep. This comes from `make collect-sim`, a simulation, not
measured data:

| LED level | Image |
|---|---|
| 0–4 | Dark and noisy; contact marks are hard to see |
| 8–12 | Brightest usable range; best contact contrast |
| 15 | Bright areas saturate at 255, so contrast drops again |

## Requirements

Python 3.10+, `h5py`, `numpy`, `opencv-python`, `matplotlib`, `digit-interface`,
installed with [uv](https://docs.astral.sh/uv/). Live capture needs the DIGIT connected over USB
(Linux: your user must have access to `/dev/video*`).

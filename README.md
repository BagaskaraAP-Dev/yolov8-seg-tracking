# segtrack

[![CI](https://github.com/BagaskaraAP-Dev/yolov8-seg-tracking/actions/workflows/ci.yml/badge.svg)](https://github.com/BagaskaraAP-Dev/yolov8-seg-tracking/actions/workflows/ci.yml)

Instance segmentation plus multi-object tracking on video: every object
gets a coloured mask, a persistent id, a motion trail, and an optional
count as it crosses a line you define.

This is a rewrite of **[YOLOv8 Segmentation with DeepSORT Object Tracking](https://github.com/MuhammadMoinFaisal/YOLOv8_Segmentation_DeepSORT_Object_Tracking)
by [Muhammad Moin Faisal](https://github.com/MuhammadMoinFaisal)**, built
on the public [ultralytics](https://github.com/ultralytics/ultralytics)
API instead of a vendored copy of it. The design is his; the code here is
new. [NOTICE](NOTICE) has the full attribution chain and the licence
history.

Indonesian walkthrough, written to be read front to back:
**[docs/CARA-KERJA.md](docs/CARA-KERJA.md)**. Colab quickstart:
**[notebooks/quickstart.ipynb](notebooks/quickstart.ipynb)**.

---

## What it does

Given a video, a camera, an image or a folder of images:

1. Finds objects and their pixel-accurate outlines with a YOLO
   segmentation model.
2. Follows each object across frames so it keeps the same id, using
   BoT-SORT or ByteTrack.
3. Draws the mask, the box, the id and a trail of where it has been, all
   in one colour per class.
4. Counts objects crossing a line, split by direction and by class.
5. Writes the result to a video file, a preview window, or both.

## Install

Python 3.10 or newer.

```bash
git clone https://github.com/BagaskaraAP-Dev/yolov8-seg-tracking.git
cd yolov8-seg-tracking
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -e .
```

Model weights are not in this repository. `ultralytics` downloads the one
you name on first use.

## Run

```bash
# Track everything in a clip and save the render
segtrack --source clip.mp4 --save out/clip-tracked.mp4

# Vehicles only, counting across a line at y=600, previewed live
segtrack --source traffic.mp4 \
         --classes car,truck,bus,motorcycle \
         --line 0,600,1280,600 \
         --show

# Webcam, CPU, boxes only, no trail
segtrack --source 0 --device cpu --no-masks --trail 0

# A model you trained yourself
segtrack --source field.mp4 --model runs/segment/train/weights/best.pt
```

`segtrack --help` lists every option. `python -m segtrack` works the same
as the installed command.

## Options

| Option | Default | What it does |
| --- | --- | --- |
| `--source` | required | video, image, folder, stream URL, or camera index (`0`) |
| `--model` | `yolov8n-seg.pt` | any ultralytics segmentation checkpoint |
| `--device` | `auto` | `auto`, `cpu`, `cuda`, `cuda:1`, `mps`, or a GPU index |
| `--classes` | all | restrict by index or name: `0,2,3` or `person,car` |
| `--tracker` | `botsort.yaml` | `botsort.yaml` survives longer occlusions, `bytetrack.yaml` is lighter |
| `--conf` / `--iou` | `0.25` / `0.7` | detection confidence and NMS thresholds |
| `--imgsz` | `640` | inference resolution |
| `--trail` | `64` | trail length in frames, `0` draws none |
| `--anchor` | `bottom` | the point representing an object: `bottom` or `center` |
| `--line` | off | counting line as `x1,y1,x2,y2` |
| `--loose-line` | off | count side changes even past the line's endpoints |
| `--no-masks` / `--mask-alpha` | on / `0.45` | mask tint |
| `--no-hud` | off | hide the status panel |
| `--watermark` / `--no-watermark` | on | authorship and licence line on rendered frames |
| `--save` / `--show` | off | write a video, open a preview window |
| `--max-frames` | all | stop early, useful for testing a line position |

### Finding the counting line

`--line` is in pixels, origin top-left. Render ten frames with
`--max-frames 10 --save probe.mp4`, open one, read the coordinates off
it, then run for real. Direction follows the order of the points:
`in` means crossing from the right of `x1,y1 -> x2,y2` to its left. Swap
the two ends to swap the labels.

## What changed from the project this descends from

Each row is a defect found by reading that repository, and what replaced it.

| Upstream behaviour | Here |
| --- | --- |
| `use_cuda=True` hard-coded, so it raised on any CPU-only machine | `--device` resolves CUDA, MPS or CPU, and falls back with a warning instead of raising |
| 13,369 lines of ultralytics 8.0.3 vendored into the repo | `ultralytics` is a pinned dependency from PyPI |
| hydra CLI, removed from ultralytics weeks after that snapshot | `argparse`, standard library |
| `deep_sort_pytorch` and its ReID weights downloaded from a third-party Google Drive folder | BoT-SORT and ByteTrack ship with `ultralytics` |
| Had to be launched from inside `ultralytics/yolo/v8/segment` because a config was loaded by relative path | runs from anywhere |
| `# TODO: filter by classes`, with no `classes` option anywhere | `--classes`, by index or by name, validated at startup |
| Vehicle counting shown in the README, absent from the code | implemented, with exact segment-intersection geometry and per-class totals |
| Mask colours from one palette, box colours from another | one colour per class, derived from the class name, used for both |
| Class colours hard-coded to COCO indices `0/2/3/5` | colour derived from the class name, so custom models work |
| Trails in a module-level dict, leaking between videos in one process | `TrailStore` instance, with `reset()` |
| Prune crashed when handed its own default `identities=None` | `prune()` takes any iterable, including an empty one |
| Label text always white, invisible on light boxes | black or white per box, chosen to clear WCAG 4.5:1 |
| Labels always drawn above the box, off-screen at the top edge | drawn inside the box when there is no room above |
| `int((y2 + y2) / 2)`, reading like a typo for the centre | `--anchor bottom\|center`, named and documented |
| Masks appended only for non-empty detections, so `masks[idx]` could index the wrong stream in a batch | mask count is checked against track count; mismatch skips the overlay and says so |
| `numpy>=1.18.5`, `torch>=1.7.0`, no upper bounds; `np.float` still called in `autobackend.py` | upper bounds pinned, no removed NumPy aliases |
| `LICENSE` deleted while every file kept its GPL-3.0 header | [LICENSE](LICENSE) present, [NOTICE](NOTICE) explains the chain |
| A Roboflow API key committed in plain text in a notebook | no credentials in this repository |
| Two notebooks totalling 26 MB of embedded output | one notebook, outputs stripped |
| No tests, no CI | 120 tests, GitHub Actions on Python 3.10, 3.12 and 3.13 |

## Layout

```
src/segtrack/
  colors.py     class name -> stable colour, and readable label text
  points.py     box -> the one point that represents the object
  trails.py     per-track position history
  counting.py   line geometry and the crossing counter
  overlay.py    mask compositing, NumPy only
  draw.py       everything that needs OpenCV
  device.py     CUDA / MPS / CPU selection
  classes.py    the class filter
  tracker.py    the pipeline: model -> tracks -> annotated frame
  cli.py        argparse entry point
tests/          120 tests, no model or video needed
docs/           CARA-KERJA.md, the walkthrough
notebooks/      quickstart.ipynb, outputs stripped
```

The split is deliberate: `colors`, `points`, `trails`, `counting`,
`overlay`, `device` and `classes` import neither OpenCV nor ultralytics,
so the logic can be tested in a second without a GPU, a checkpoint or a
video file.

## Tests

```bash
pip install -e ".[dev]"
pytest
ruff check .
```

120 tests. 110 of them need neither OpenCV nor ultralytics installed,
and the whole suite finishes in about 0.3 seconds, because the logic
modules are kept free of both. They cover colour distinctness and label
contrast, trail lifetime, counting geometry, mask compositing, device
fallback and class-filter parsing. The remaining 10 cover the CLI wiring
and need OpenCV, so they skip rather than fail on a partial install.

Nothing in the suite runs inference, so it needs no weights and no GPU.

**Not covered by tests:** the frames that come back from
`model.track()`. Mask-to-track index alignment, the writer and the
preview window are exercised by running the tool, not by the suite. If
you hit a case where masks and ids disagree, the pipeline logs it and
skips the overlay rather than painting the wrong mask on the wrong
object.

No speed figures are quoted here because none have been measured on
hardware worth quoting. Throughput depends on the checkpoint, the
resolution and the device; measure it with `--max-frames` on your own
machine.

## Licence and credit

AGPL-3.0, because it links against `ultralytics`, which is AGPL-3.0. See
[LICENSE](LICENSE).

Copyright (C) 2026 Bagaskara Amukti Palapa.

Rendered video carries the same line in its bottom-right corner, so a
clip that travels away from this repository still says who made it and
under what licence. `--no-watermark` turns it off; `--watermark "text"`
replaces it.

Credit for the original design, the Colab notebooks and the video
tutorials belongs to [Muhammad Moin Faisal](https://github.com/MuhammadMoinFaisal),
and for YOLO and the `ultralytics` package to
[Ultralytics](https://github.com/ultralytics/ultralytics). Neither
endorses this rewrite. [NOTICE](NOTICE) records what came from where.

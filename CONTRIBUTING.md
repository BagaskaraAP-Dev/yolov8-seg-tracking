# Contributing

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Before pushing

```bash
ruff check .
pytest -q
```

Both run in CI on every push, so a failure here is a failure there.

## Where new code goes

If a piece of logic can be written without OpenCV and without
ultralytics, put it in its own module and give it tests. That is what
keeps most of the suite running in under a second with no GPU, no
checkpoint and no video file, and it is what caught the colour-collision
bug described in `docs/CARA-KERJA.md` before it ever reached a frame.

`draw.py` is the only module that imports `cv2`. `tracker.py` is the
only one that imports `ultralytics`, and it does so inside a function so
that `--help` and the test suite do not pay for loading torch.

## Two rules this repository exists to uphold

**No credentials, ever.** The project this descends from shipped a
Roboflow API key in plain text inside a committed notebook. Deleting
such a file does not help: the key stays in git history and has to be
revoked. Keep keys in environment variables.

**Strip notebook output before committing.** Those same upstream
notebooks were 10 MB and 16 MB because the rendered frames were embedded
in the JSON, which is most of why that repository is 35 MB.

```bash
pip install nbstripout
nbstripout notebooks/*.ipynb
```

## Attribution

Keep `NOTICE` accurate. The design of this project is Muhammad Moin
Faisal's, and the removal of that credit in an intermediate fork is the
reason the file is written the way it is. If you add code derived from
someone else's work, add them to it.

## Licensing of contributions

By opening a pull request you agree your contribution is licensed
AGPL-3.0-or-later, matching the rest of the project.

# segtrack - YOLO instance segmentation with multi-object tracking
# Copyright (C) 2026 Bagaskara Amukti Palapa
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Rewritten from the GPL-3.0 project by Muhammad Moin Faisal; see NOTICE.

"""Parsing for the class filter.

The project this replaces left a "TODO: filter by classes" in its
segmentation predictor and exposed no classes key in its config, so
there was no way to track only vehicles or only people. The filter is
the most useful knob on a tracking demo, so it is wired here, and it
accepts names as well as indices.
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = ["parse_classes"]

ClassSpec = str | Sequence[str | int] | None


def parse_classes(spec: ClassSpec, names: dict[int, str] | None = None) -> list[int] | None:
    """Resolve a class filter into sorted class indices.

    None or an empty value means every class, which is also how
    ultralytics expresses it. Entries may be indices ("2") or class
    names ("car"); names are matched case-insensitively against names,
    the model's own mapping from ``model.names``.

    Raises ValueError on an unknown name or an out-of-range index, so a
    typo fails at startup instead of silently tracking nothing.
    """
    if spec is None:
        return None

    if isinstance(spec, str):
        tokens = [token.strip() for token in spec.replace(";", ",").split(",")]
    else:
        tokens = [str(token).strip() for token in spec]
    tokens = [token for token in tokens if token]

    if not tokens:
        return None

    by_name = {label.lower(): index for index, label in (names or {}).items()}
    resolved = set()

    for token in tokens:
        if token.lstrip("-").isdigit():
            index = int(token)
            if index < 0:
                raise ValueError(f"class index {index} is negative")
            if names is not None and index not in names:
                raise ValueError(
                    f"class index {index} is not in this model, which has "
                    f"{len(names)} classes (0-{max(names)})"
                )
            resolved.add(index)
            continue

        if token.lower() in by_name:
            resolved.add(by_name[token.lower()])
            continue

        if names is None:
            raise ValueError(
                f"class name {token!r} cannot be resolved before the model is loaded"
            )
        known = ", ".join(sorted(names.values()))
        raise ValueError(f"unknown class name {token!r}; this model knows: {known}")

    return sorted(resolved)

# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Aleksander Grochowicz
#
# SPDX-License-Identifier: MIT
"""
Generate slack-sweep config files from a base config.

Reads ``config/slack_sweep.yaml`` (a base config path + a set of sweep entries,
each mapping a name to an absolute slack value and a human-readable label) and
writes one standalone config per entry: ``config/<name>.yaml``.

Each generated config is the base config copied verbatim, with exactly two lines
overridden:

  - ``run.prefix``            -> the sweep entry name
  - ``near-opt.slack.value``  -> the entry's slack value (with the label as comment)

Everything else (comments, ordering, formatting) is preserved byte-for-byte,
because the base and each sweep config differ only in those two lines. Both
target lines are unique in the base config, so a targeted line replacement is
unambiguous.

Usage
-----
    python scripts/generate_slack_sweep_configs.py
"""

import re
import sys
from pathlib import Path

import yaml

# Anchors: both patterns match exactly one line in the base config.
PREFIX_RE = re.compile(r"^  prefix:.*$", re.MULTILINE)
SLACK_VALUE_RE = re.compile(r"^    value:.*$", re.MULTILINE)


def render_config(base_text: str, name: str, value: str, label: str) -> str:
    """Return base_text with run.prefix and near-opt.slack.value overridden."""
    new_text, n_prefix = PREFIX_RE.subn(f'  prefix: "{name}"', base_text)
    if n_prefix != 1:
        raise ValueError(
            f"Expected exactly one 'prefix:' line in base config, found {n_prefix}"
        )
    new_text, n_slack = SLACK_VALUE_RE.subn(f"    value: {value} # {label}", new_text)
    if n_slack != 1:
        raise ValueError(
            f"Expected exactly one near-opt slack 'value:' line, found {n_slack}"
        )
    return new_text


def main() -> None:
    config_dir = Path(__file__).resolve().parents[1] / "config"
    sweep_file = config_dir / "slack_sweep.yaml"

    with open(sweep_file) as f:
        spec = yaml.safe_load(f)

    base_path = Path(__file__).resolve().parents[1] / spec["base"]
    base_text = base_path.read_text()

    for name, entry in spec["sweep"].items():
        value, label = str(entry["value"]), str(entry["label"])
        out_path = config_dir / f"{name}.yaml"
        out_path.write_text(render_config(base_text, name, value, label))
        print(f"wrote {out_path.relative_to(config_dir.parent)}  (slack={value}, {label})")


if __name__ == "__main__":
    main()

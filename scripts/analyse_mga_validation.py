# SPDX-FileCopyrightText: 2026 Aleksander Grochowicz
#
# SPDX-License-Identifier: MIT

"""
Post-hoc analysis of MGA validation runs. No network loading.

Reads test_mga_operations outputs per (d, direction_hash, w) and produces:
  - mga_validation_atomic_{wildcards}.csv   one row per (d, c, w): shedding, δ,
                                            prices, emissions, costs, slack_spent
  - mga_validation_rollup_{wildcards}.csv   one row per (d, c): S, BCR, worst_w
"""

import json
import logging
import re
from pathlib import Path

import pandas as pd

import resilience_analysis_helpers as rh
from _helpers import configure_logging
from analyse_cost_optimal_validation import load_val_outputs

logger = logging.getLogger(__name__)


def analyse_mga_validation(
    val_triplets: list[dict],
    baseline_shedding_path: str,
    cost_opt_summary_path: str,
    cache_dir: str,
    directions_lookup: dict,
    output_atomic: str,
    output_rollup: str,
):
    Path(output_atomic).parent.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(cache_dir)

    baseline = pd.read_csv(baseline_shedding_path).set_index(["d", "w"])
    with open(cost_opt_summary_path) as f:
        cost_opt_total = json.load(f)["total_cost"]
    # keyed by design year (d); all candidates share the same cost-optimal reference
    baseline_totals = {t["d"]: cost_opt_total for t in val_triplets}

    rows = []
    for t in val_triplets:
        d, dh, nh, w = t["d"], t["direction_hash"], t["network_hash"], t["w"]
        val_dir = Path(t["val_dir"])

        data = load_val_outputs(val_dir)
        shed = rh.shedding_stats(data["load_shed"], data["heat_shed"])

        base = baseline.loc[(d, w)]
        delta = rh.delta_shed_cost(shed, float(base["shed_cost"]))

        info_path = cache_dir / "info" / f"info_{nh}_{dh}.json"
        info = json.loads(info_path.read_text())
        slack_spent = info["total_cost"] - baseline_totals[d]

        direction = directions_lookup[dh]

        def price_stats(s: pd.Series, name: str) -> dict:
            return {f"{name}_mean": float(s.mean()),
                    f"{name}_p50":  float(s.quantile(0.5)),
                    f"{name}_p90":  float(s.quantile(0.9))}

        rows.append({
            "d": d, "direction_hash": dh, "w": w,
            **{f"v_{k}": v for k, v in direction.items()},
            **shed,
            "delta_shed_cost":  delta,
            "emissions_total":  float(data["emissions"].sum()),
            "net_negative":     float(data["emissions"].sum()) <= 0.0,
            **price_stats(data["elec_price"], "elec_price"),
            **price_stats(data["heat_price"], "heat_price"),
            **price_stats(data["h2_price"],   "h2_price"),
            **price_stats(data["co2_price"],  "co2_price"),
            "capex":       info["capex"],
            "opex":        info["opex"],
            "total_cost":  info["total_cost"],
            "slack_spent": slack_spent,
        })
        logger.info(f"d={d} dh={dh[:8]}… w={w} delta={delta:.2e}")

    atomic = pd.DataFrame(rows)
    atomic.to_csv(output_atomic, index=False)

    v_cols = [c for c in atomic.columns if c.startswith("v_")]
    rollup_rows = []

    for (d, dh), grp in atomic.groupby(["d", "direction_hash"]):
        deltas = grp["delta_shed_cost"].values
        S, worst_idx = rh.configuration_score(deltas)
        cv, _ = rh.cvar(deltas)
        worst_w = grp.iloc[grp["delta_shed_cost"].values.argmin()]["w"]

        slack = grp["slack_spent"].iloc[0]
        mean_delta = float(deltas.mean())
        b = rh.bcr(mean_delta, slack)

        rollup_rows.append({
            "d": d, "direction_hash": dh,
            **grp[v_cols].iloc[0].to_dict(),
            "S":          S,
            "worst_w":    worst_w,
            "mean_delta": mean_delta,
            "cvar_delta": cv,
            "BCR":        b,
        })

    pd.DataFrame(rollup_rows).to_csv(output_rollup, index=False)
    logger.info(
        f"{len(rows)} triples, {len(rollup_rows)} candidates written"
    )


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "analyse_mga_validation",
            opts="",
            clusters="50",
            sector_opts="",
            planning_horizons="weather_year_1941_3H",
            configfiles="config/test/config.overnight.yaml",
        )

    configure_logging(snakemake)

    network_hash = Path(snakemake.input.network_hash).read_text().strip()

    near_opt = pd.read_csv(snakemake.input.near_opt_solutions)
    dir_cols = [c for c in near_opt.columns if c.startswith("dir_") and c != "dir_hash"]
    dims = [c[len("dir_"):] for c in dir_cols]
    directions_lookup = {
        row["dir_hash"]: {dim: row[f"dir_{dim}"] for dim in dims}
        for _, row in near_opt.iterrows()
    }

    val_triplets = []
    files = snakemake.input.load_shedding
    if isinstance(files, str):
        files = [files]

    for f in files:
        m = re.search(
            r"/([^/]+)/validation/mga_([a-f0-9]+)_([a-f0-9]+)_(weather_year_\d+_\d+H)_base_s_",
            str(f)
        )
        if m:
            d, config_hash, direction_hash, w = m.groups()
            val_triplets.append({
                "d": d,
                "config_hash":    config_hash,
                "direction_hash": direction_hash,
                "network_hash":   network_hash,
                "w": w,
                "val_dir": str(Path(f).parent),
            })

    analyse_mga_validation(
        val_triplets=val_triplets,
        baseline_shedding_path=snakemake.input.baseline_shedding,
        cost_opt_summary_path=snakemake.input.cost_opt_summary,
        cache_dir=snakemake.params.cache_dir,
        directions_lookup=directions_lookup,
        output_atomic=snakemake.output.atomic,
        output_rollup=snakemake.output.rollup,
    )

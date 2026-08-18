# SPDX-FileCopyrightText: 2026 Aleksander Grochowicz
#
# SPDX-License-Identifier: MIT

"""
Analyse cost-optimal validation runs to produce baseline shedding statistics.

Reads test_operations outputs directly (no network loading) and produces
two artifacts per design year:
  - baseline_shedding_{network_hash}.csv  (handoff to analyse_mga_validation)
  - baseline_analysis_{network_hash}.csv  (full stats for plotting)
"""

import json
from pathlib import Path

import pandas as pd

import resilience_analysis_helpers as rh
from _helpers import configure_logging


def load_val_outputs(val_dir: Path) -> dict:
    """
    Load all test_operations outputs from one validation directory.

    Parameters
    ----------
    val_dir : Path
        Directory containing load_shedding.csv, heat_shedding.csv, etc.

    Returns
    -------
    dict with keys: load_shed, heat_shed, net_load, emissions,
                    elec_price, heat_price, h2_price, co2_price, objective.
    All price/shedding series are spatially aggregated (sum or mean across nodes).
    """
    return {
        "load_shed":  pd.read_csv(val_dir / "load_shedding.csv",  index_col=0).sum(axis=1),
        "heat_shed":  pd.read_csv(val_dir / "heat_shedding.csv",  index_col=0).sum(axis=1),
        "net_load":   pd.read_csv(val_dir / "net_load.csv",       index_col=0),
        "emissions":  pd.read_csv(val_dir / "emissions.csv",      index_col=0).squeeze(),
        "elec_price": pd.read_csv(val_dir / "elec_prices.csv",    index_col=0).mean(axis=1),
        "heat_price": pd.read_csv(val_dir / "heat_prices.csv",    index_col=0).mean(axis=1),
        "h2_price":   pd.read_csv(val_dir / "h2_prices.csv",      index_col=0).mean(axis=1),
        "co2_price":  pd.read_csv(val_dir / "co2_prices.csv",     index_col=0).mean(axis=1),
        "objective":  json.loads((val_dir / "objective.json").read_text()),
    }


def compute_val_stats(d: str, w: str, data: dict) -> dict:
    """
    Compute validation statistics for one (design year, stress year) pair.

    Parameters
    ----------
    d : str  Design year identifier.
    w : str  Stress/validation year identifier.
    data : dict  Output of load_val_outputs.

    Returns
    -------
    Flat dict of scalars: shedding stats, price percentiles, emissions, costs.
    """
    def price_stats(s: pd.Series, name: str) -> dict:
        return {
            f"{name}_mean": float(s.mean()),
            f"{name}_p50":  float(s.quantile(0.5)),
            f"{name}_p90":  float(s.quantile(0.9)),
            f"{name}_p99":  float(s.quantile(0.99)),
        }

    em_total = float(data["emissions"].sum())

    return {
        "d": d, "w": w,
        **rh.shedding_stats(data["load_shed"], data["heat_shed"]),
        **price_stats(data["elec_price"], "elec_price"),
        **price_stats(data["heat_price"], "heat_price"),
        **price_stats(data["h2_price"],   "h2_price"),
        **price_stats(data["co2_price"],  "co2_price"),
        "emissions_total": em_total,
        "net_negative":    em_total <= 0.0,
        "capex":           data["objective"]["capex"],
        "opex":            data["objective"]["opex"],
        "total_cost":      data["objective"]["total"],
    }



def analyse_cost_optimal_val(
    val_dirs: dict[tuple, Path],
) -> pd.DataFrame:
    """
    Analyse all (d, w) validation pairs.

    Returns
    -------
    DataFrame with one row per (d, w) containing shedding stats,
    price percentiles, emissions, and costs.
    """
    rows = [compute_val_stats(d, w, load_val_outputs(val_dir))
            for (d, w), val_dir in val_dirs.items()]
    return pd.DataFrame(rows)

if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "analyse_cost_optimal_validation",
            opts="",
            clusters="50",
            sector_opts="",
            planning_horizons="weather_year_1941_3H",
            configfiles="config/test/config.overnight.yaml",
        )

    configure_logging(snakemake)

    wc = snakemake.wildcards
    config_str = f"base_s_{wc.clusters}_{wc.opts}_{wc.sector_opts}_{wc.planning_horizons}"

    val_dirs = {}
    for obj_path in snakemake.input.val_dirs:
        val_dir = Path(obj_path).parent
        w = val_dir.name.replace(f"_{config_str}", "")
        val_dirs[(wc.run, w)] = val_dir

    network_hash = Path(snakemake.input.network_hash).read_text().strip()

    df = analyse_cost_optimal_val(val_dirs)

    shedding_cols = ["d", "w", "load_shed_mwh", "heat_shed_mwh", "shed_cost"]
    df[shedding_cols].to_csv(snakemake.output.baseline_shedding, index=False)
    df.to_csv(snakemake.output.baseline_analysis, index=False)
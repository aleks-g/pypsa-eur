# SPDX-FileCopyrightText: 2026 Aleksander Grochowicz
#
# SPDX-License-Identifier: MIT

"""
Characterise MGA candidates from cache files. No network loading.

Reads per-candidate cache files (caps, info, netload, emissions) and the
cost-optimal reference artifacts, and produces a single CSV with one row
per candidate containing capacity shares, concentration metrics, slack,
net load stats, and emissions profile match.

Outputs:
  {run}/resilience/mga_candidates_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

import resilience_analysis_helpers as rh
from _helpers import configure_logging


def load_mga_cache(cache_dir: Path, network_hash: str, direction_hash: str) -> dict:
    """
    Load all four cache files for one MGA candidate.

    Parameters
    ----------
    cache_dir : Path  Root of the MGA cache directory.
    network_hash : str  8-character network hash.
    direction_hash : str  16-character direction hash.

    Returns
    -------
    dict with keys: caps (DataFrame), info (dict), netload (DataFrame),
                    emissions (Series).
    """
    nh, dh = network_hash, direction_hash
    return {
        "caps":      pd.read_csv(cache_dir / "caps"      / f"caps_{nh}_{dh}.csv"),
        "info":      json.loads((cache_dir / "info"      / f"info_{nh}_{dh}.json").read_text()),
        "netload":   pd.read_csv(cache_dir / "netload"   / f"netload_{nh}_{dh}.csv", index_col=0),
        "emissions": pd.read_csv(cache_dir / "emissions" / f"emissions_{nh}_{dh}.csv", index_col=0).squeeze(),
    }


def characterise_candidate(
    cache: dict,
    direction: dict,
    opt_caps: pd.DataFrame,
    opt_emissions: pd.Series,
    opt_total_cost: float,
    carrier_lookup: dict,
    tech_categories: dict,
    capital_costs: pd.Series | None = None,
) -> dict:
    """
    Compute all characterisation stats for one MGA candidate.

    Parameters
    ----------
    cache : dict            Output of load_mga_cache.
    direction : dict        Direction vector keyed by DIRECTION_DIMS.
    opt_caps : pd.DataFrame Cost-optimal caps (parsed, with carrier/tech/node cols).
    opt_emissions : pd.Series  Cost-optimal hourly emissions.
    opt_total_cost : float  Cost-optimal total system cost (for slack calculation).
    carrier_lookup : dict   Output of rh.build_carrier_lookup.
    tech_categories : dict  Projection config tech categories.

    Returns
    -------
    Flat dict of scalars — one row in the output CSV.
    """
    info = cache["info"]

    # Parse and aggregate capacities
    caps = rh.parse_caps(cache["caps"], carrier_lookup)
    conc = rh.concentration_stats(caps, tech_categories, capital_costs=capital_costs)
    jsds = rh.jsd_vs_reference(caps, opt_caps, tech_categories, capital_costs=capital_costs)

    # Slack
    slack_total = info["total_cost"] - opt_total_cost

    # Net load stats
    nl_stats = {}
    for col in ["elec", "heat"]:
        if col in cache["netload"].columns:
            s = cache["netload"][col]
            nl_stats[f"netload_peak_{col}"]        = float(s.max())
            nl_stats[f"netload_top100_mean_{col}"] = float(s.nlargest(100).mean())

    # Emissions profile match vs cost-optimal
    emis_match = rh.profile_match(cache["emissions"], opt_emissions)

    return {
        "network_hash":   info["network_hash"],
        "direction_hash": info["direction_hash"],
        "design_year":    info.get("design_year", ""),
        **{f"v_{k}": v for k, v in direction.items()},
        "capex":          info["capex"],
        "opex":           info["opex"],
        "total_cost":     info["total_cost"],
        "slack_total":    slack_total,
        **conc,
        **jsds,
        **nl_stats,
        **emis_match,
    }


def analyse_mga_candidates(
    cache_dir: Path,
    directions_lookup: dict,
    opt_caps: pd.DataFrame,
    opt_emissions: pd.Series,
    opt_total_cost: float,
    carrier_lookup: dict,
    tech_categories: dict,
    candidates: list[dict],
    capital_costs: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Characterise all MGA candidates.

    Parameters
    ----------
    candidates : list of dicts with keys network_hash, direction_hash.
    directions_lookup : dict mapping direction_hash → direction vector dict.

    Returns
    -------
    DataFrame with one row per candidate.
    """
    rows = []
    for cand in candidates:
        nh, dh = cand["network_hash"], cand["direction_hash"]
        cache = load_mga_cache(cache_dir, nh, dh)
        direction = directions_lookup[dh]
        rows.append(characterise_candidate(
            cache=cache,
            direction=direction,
            opt_caps=opt_caps,
            opt_emissions=opt_emissions,
            opt_total_cost=opt_total_cost,
            carrier_lookup=carrier_lookup,
            tech_categories=tech_categories,
            capital_costs=capital_costs,
        ))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "analyse_mga_candidates",
            opts="",
            clusters="50",
            sector_opts="",
            planning_horizons="2050",
            run="weather_year_1941_3H",
            configfiles="config/test/config.overnight.yaml",
        )

    configure_logging(snakemake)

    wc = snakemake.wildcards
    tech_categories = {
        tech: [entry["carrier"] for entry in entries]
        for tech, entries in snakemake.config["near-opt"]["projection"].items()
    }
    carrier_lookup = rh.build_carrier_lookup(tech_categories)

    # Load cost-optimal reference artifacts
    opt_caps = rh.parse_caps(
        pd.read_csv(snakemake.input.cost_opt_caps),
        carrier_lookup,
    )
    with open(snakemake.input.cost_opt_summary) as f:
        opt_summary = json.load(f)
    opt_total_cost = opt_summary["total_cost"]
    opt_emissions = pd.read_csv(
        snakemake.input.cost_opt_emissions, index_col=0
    ).squeeze()

    capital_costs = (
        pd.read_csv(snakemake.input.capital_costs)
        .drop_duplicates(subset="name", keep="first")
        .set_index("name")["capital_cost"]
    )

    # Build direction lookup from near_opt_solutions CSV
    near_opt = pd.read_csv(snakemake.input.near_opt_solutions)
    dir_cols = [c for c in near_opt.columns if c.startswith("dir_") and c != "dir_hash"]
    dims = [c[len("dir_"):] for c in dir_cols]
    directions_lookup = {
        row["dir_hash"]: {dim: row[f"dir_{dim}"] for dim in dims}
        for _, row in near_opt.iterrows()
    }

    # Build candidate list from info files (filter to only cache info JSONs)
    candidates = [
        {"network_hash": Path(f).stem.split("_")[1],
         "direction_hash": "_".join(Path(f).stem.split("_")[2:])}
        for f in snakemake.input.info_files
        if Path(f).name.startswith("info_") and Path(f).suffix == ".json"
    ]

    df = analyse_mga_candidates(
        cache_dir=Path(snakemake.params.cache_dir),
        directions_lookup=directions_lookup,
        opt_caps=opt_caps,
        opt_emissions=opt_emissions,
        opt_total_cost=opt_total_cost,
        carrier_lookup=carrier_lookup,
        tech_categories=tech_categories,
        candidates=candidates,
        capital_costs=capital_costs,
    )

    df.to_csv(snakemake.output.candidates, index=False)
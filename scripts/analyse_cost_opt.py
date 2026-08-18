# SPDX-FileCopyrightText: 2026 Aleksander Grochowicz
#
# SPDX-License-Identifier: MIT

"""
Analyse cost-optimal network to produce design-year characterisation artifacts.

Reads the solved network directly (the only script that does so) and produces:
  - cost_opt_summary_{wildcards}.json    scalars: total_cost, capex, opex, co2_shadow_price
  - cost_opt_caps_{wildcards}.csv        optimal capacities (component, name, attribute, value)
  - cost_opt_netload_{wildcards}.csv     net load stats (elec + heat)
  - cost_opt_prices_{wildcards}.csv      marginal price percentiles (elec, heat)
"""

import json
from pathlib import Path

import pandas as pd
import pypsa

import resilience_analysis_helpers as rh
from mga_helpers import extract_objective, extract_emissions, extract_optimal_capacities
from _helpers import configure_logging


def extract_capital_costs(n: pypsa.Network) -> pd.DataFrame:
    """
    Extract capital cost per component name from the solved network.

    Returns
    -------
    DataFrame with columns: name, capital_cost (EUR/MW or EUR/MWh).
    """
    rows = []
    _comp_attr = {
        "Generator": "generators",
        "StorageUnit": "storage_units",
        "Store": "stores",
        "Link": "links",
        "Line": "lines",
    }
    for comp, attr in _comp_attr.items():
        df = getattr(n, attr)
        if "capital_cost" in df.columns:
            for name, cc in df["capital_cost"].items():
                rows.append({"name": name, "capital_cost": float(cc)})
    return pd.DataFrame(rows)


def analyse_cost_opt(n: pypsa.Network) -> dict:
    """
    Extract all cost-optimal characterisation artifacts from a solved network.

    Returns
    -------
    dict with keys: summary (dict), caps (DataFrame),
                    netload_stats (DataFrame), price_stats (DataFrame).
    """
    obj = extract_objective(n)

    co2_shadow = None
    try:
        for name in ["CO2Limit", "co2_limit", "CO2"]:
            if name in n.global_constraints.index:
                co2_shadow = float(n.global_constraints.at[name, "mu"])
                break
    except Exception:
        pass

    return {
        "summary": {
            "total_cost":       obj["total"],
            "capex":            obj["capex"],
            "opex":             obj["opex"],
            "co2_shadow_price": co2_shadow,
        },
        "caps":          extract_optimal_capacities(n),
        "netload_stats": rh.netload_stats(n),
        "price_stats":   rh.price_stats(n),
        "emissions":     extract_emissions(n),
    }


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "analyse_cost_opt",
            opts="",
            clusters="50",
            sector_opts="",
            planning_horizons="weather_year_1941_3H",
            configfiles="config/test/config.overnight.yaml",
        )

    configure_logging(snakemake)

    n = pypsa.Network(snakemake.input.network)
    results = analyse_cost_opt(n)

    with open(snakemake.output.summary, "w") as f:
        json.dump(results["summary"], f, indent=2)

    results["caps"].to_csv(snakemake.output.caps, index=False)
    results["netload_stats"].to_csv(snakemake.output.netload_stats, index=False)
    results["price_stats"].to_csv(snakemake.output.price_stats, index=False)
    results["emissions"].to_frame(name="co2_flow").to_csv(snakemake.output.emissions)
    extract_capital_costs(n).to_csv(snakemake.output.capital_costs, index=False)
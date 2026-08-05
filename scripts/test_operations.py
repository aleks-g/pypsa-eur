# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2023-2025 Aleksander Grochowicz & Koen van Greevenbroek
#
# SPDX-License-Identifier: MIT
"""
Solves linear optimal dispatch in hourly resolution using the capacities of
previous capacity expansion in rule `solve_network`.
"""

import json
import logging

import numpy as np
import pandas as pd
import pypsa
import sys
from _helpers import (
    configure_logging,
    set_scenario_config,
    update_config_from_wildcards,
)
from solve_network import prepare_network, collect_kwargs, create_optimization_model
from _benchmark import memory_logger
from mga_helpers import extract_net_load, extract_emissions, extract_objective

logger = logging.getLogger(__name__)


def set_weather(
    n: pypsa.Network,
    n_weather: pypsa.Network,
) -> None:
    """Set weather-dependent parameters from n_weather to n."""
    for c, attr in [
        ("Generator", "p_max_pu"),
        ("StorageUnit", "p_max_pu"),
        ("StorageUnit", "inflow"),
        ("Load", "p_set"),
        ("Link", "efficiency")
    ]:
        target = n.pnl(c)[attr]
        source = n_weather.pnl(c)[attr].values

        # Check if the source has the same shape as the target
        if target.shape != source.shape:
            logger.error(
                f"Shape mismatch between source and target for {c} {attr}: "
                f"{source.shape} != {target.shape}"
            )
            raise ValueError(
                f"Shape mismatch between source and target for {c} {attr}: "
                f"{source.shape} != {target.shape}"
            )
        else:
            target.loc[:, :] = source

def set_co2_price(
    n: pypsa.Network,
) -> None:
    """Sets CO2 price based on the dual variable of the CO2 constraint (from the already optimized network n). Removes CO2 limit."""
    # Follow implementation roughly by Gotske et al, 2024. (https://github.com/ebbekyhl/multi-weather-year-assessment/blob/8aed88728e7a0848de5fd987ff8303761a8f5687/scripts/update_network.py#L150)

    # Extract CO2 price from optimised network
    co2_price = -n.global_constraints.loc["CO2Limit","mu"] # in EUR/tCO2

    # Remove hard CO2 cap.
    n.remove("GlobalConstraint", "CO2Limit")

    # Add CO2 price to all emitters - note that net removers will have marginal prices lowered by CO2 price.
    # All is weighted by efficiency.
    # bus1: Process emissions, HVC
    process_i = n.links.query('bus1 == "co2 atmosphere"').index
    process = n.links.loc[process_i]
    process_co2_price = co2_price * process.efficiency
    # bus2: power plants, industry, boilers, but also DAC, biomass to liquid etc.
    emitters_i = n.links.query('bus2 == "co2 atmosphere"').index
    emitters = n.links.loc[emitters_i]
    emitters_co2_price = co2_price * emitters.efficiency2 # note that also negative values for net removers are allowed here
    # bus3: chp
    chp_i = n.links.query('bus3 == "co2 atmosphere"').index
    chp = n.links.loc[chp_i]
    chp_co2_price = co2_price * chp.efficiency3

    # Update marginal costs
    n.links.loc[process_i, "marginal_cost"] += process_co2_price
    n.links.loc[emitters_i, "marginal_cost"] += emitters_co2_price
    n.links.loc[chp_i, "marginal_cost"] += chp_co2_price


def extract_shedding_metrics(n: pypsa.Network) -> tuple:
    """
    Extract load and heat shedding time series from solved network.

    Returns
    -------
    load_shedding : pd.DataFrame
        Load shedding time series (excluding battery and H2)
    heat_shedding : pd.DataFrame
        Heat shedding time series
    """
    # Load shedding
    load_shedding = n.generators_t.p.filter(like="load shedding", axis="columns")
    load_shedding = load_shedding.loc[
        :, ~load_shedding.columns.str.contains("battery|H2")
    ].round(0)

    # Heat shedding
    heat_shedding = n.generators_t.p.filter(like="heat shedding", axis="columns")

    return load_shedding, heat_shedding


def extract_marginal_prices(n):
    """
    Extract nodal marginal prices from a solved validation dispatch.

    Returns prices for electricity, H2, CO2, and heat variants.
    Prices are shadow prices of the node balance constraint (EUR/MWh or EUR/t for CO2).

    Returns
    -------
    dict
        Keys: 'electricity', 'h2', 'co2_stored', 'heat_urban_central',
              'heat_urban_decentral', 'heat_rural'.
        Each value is a DataFrame of marginal prices (snapshots x nodes).
    """
    prices = {}

    # Carrier-based buses
    carriers = {
        'electricity': 'AC',
        'h2': 'H2',
        'co2_stored': 'co2 stored',
    }

    for key, carrier in carriers.items():
        buses = n.buses[n.buses.carrier == carrier].index
        if not buses.empty:
            prices[key] = n.buses_t.marginal_price[buses].sort_index(axis=1)
        else:
            logger.warning(f"No buses found for '{key}'")

    # Heat buses (by name pattern)
    heat_patterns = {
        'heat_urban_central': 'urban central heat',
        'heat_urban_decentral': 'urban decentral heat',
        'heat_rural': 'rural heat',
    }

    for key, pattern in heat_patterns.items():
        buses = n.buses[n.buses.index.str.contains(pattern, case=False, na=False)].index
        if not buses.empty:
            prices[key] = n.buses_t.marginal_price[buses].sort_index(axis=1)
        else:
            logger.warning(f"No buses found for '{key}'")

    return prices

if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "test_operations",
            configfiles="config/test_sec.yaml",
            opts="50seg-lv1.0",
            clusters="50",
            sector_opts="Co2L0.0+T+H+B+I+A",
            planning_horizons="2050",
        )

    configure_logging(snakemake)
    set_scenario_config(snakemake)
    update_config_from_wildcards(snakemake.config, snakemake.wildcards)

    solve_opts = snakemake.params.solving["options"]

    # Activate load shedding
    solve_opts["load_shedding"] = True

    np.random.seed(solve_opts.get("seed", 123))

    n = pypsa.Network(snakemake.input.network)
    m = pypsa.Network(snakemake.input.weather_network)

    planning_horizons = snakemake.wildcards.get("planning_horizons", "")

    # Add small buffer to optimal capacities to avoid numerical infeasibilities
    # when running under a different weather year. Scales with retry attempt.
    capacity_buffers = [0.001, 0.005, 0.01, 0.02]
    attempt = getattr(snakemake, "attempt", 1)
    buffer = capacity_buffers[min(attempt - 1, len(capacity_buffers) - 1)]
    logger.info(f"Adding {buffer*100:.2f}% capacity buffer (attempt {attempt})")
    for comp in [n.generators, n.links, n.stores, n.storage_units]:
        for attr in ["p_nom_opt", "e_nom_opt"]:
            if attr in comp.columns:
                comp[attr] *= (1 + buffer)

    try:
        set_weather(n, m)
        n.optimize.fix_optimal_capacities()
        prepare_network(
            n,
            solve_opts=solve_opts,
            foresight=snakemake.params.foresight,
            planning_horizons=planning_horizons,
            co2_sequestration_potential=snakemake.params["co2_sequestration_potential"],
            limit_max_growth=snakemake.params.get("sector", {}).get("limit_max_growth"),
        )
        logging_frequency = snakemake.config.get("solving", {}).get(
            "mem_logging_frequency", 30
        )
        if snakemake.config["run"]["stress_tests"].get("mode", "") == "co2-price":
            print("Setting CO2 price based on previous optimization.")
            set_co2_price(n)
        with memory_logger(
            filename=getattr(snakemake.log, "memory", None), interval=logging_frequency
        ) as mem:
            model_kwargs, solve_kwargs = collect_kwargs(
                snakemake.config,
                snakemake.params.solving,
                planning_horizons,
                log_fn=snakemake.log.solver,
                mode="single",
            )
            create_optimization_model(
                n,
                config=snakemake.config,
                params=snakemake.params,
                model_kwargs=model_kwargs,
                solve_kwargs=solve_kwargs,
                planning_horizons=planning_horizons,
            )
            status, condition = n.optimize.solve_model(**solve_kwargs)
            if status == "warning":
                logger.warning(
                    f"Solver status: {status}, condition: {condition}"
                )
                try:
                    n.model.print_infeasibilities()
                except AttributeError:
                    logger.warning("print_infeasibilities not available in this pypsa version")

        logger.info(f"Maximum memory usage: {mem.mem_usage}")

        n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
        n.export_to_netcdf(snakemake.output.network)

        # Extract shedding metrics
        load_shedding, heat_shedding = extract_shedding_metrics(n)

        # Export the results
        load_shedding.round(3).to_csv(snakemake.output.load_shedding)
        heat_shedding.round(3).to_csv(snakemake.output.heat_shedding)

        # Extract net load
        nl = extract_net_load(n, heating=True)
        netload_df = pd.DataFrame({"elec": nl["elec"], "heat": nl["heat"]})
        netload_df.to_csv(snakemake.output.net_load)

        # Extract emissions
        emissions = extract_emissions(n)
        emissions.to_frame(name="co2_flow").to_csv(snakemake.output.emissions)

        # Extract prices
        prices = extract_marginal_prices(n)
        prices["electricity"].to_csv(snakemake.output.elec_prices)
        prices["h2"].to_csv(snakemake.output.h2_prices)
        prices["co2_stored"].to_csv(snakemake.output.co2_prices)

        # Combine and save all heat types
        heat_prices = pd.concat([prices[k] for k in prices if k.startswith('heat_')], axis=1)
        heat_prices.to_csv(snakemake.output.heat_prices)

        # Extract objective value
        obj = extract_objective(n)
        with open(snakemake.output.objective, "w") as f:
            json.dump(obj, f, indent=2)

        with open(snakemake.output.metadata, "w") as f:
            json.dump({"attempt": attempt, "buffer": buffer, "status": status, "condition": condition}, f, indent=2)

    except Exception as e:
        logger.exception(f"Error in test_operations: {e}")
        sys.exit(1)

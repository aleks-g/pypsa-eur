# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2023-2025 Aleksander Grochowicz
#
# SPDX-License-Identifier: MIT
"""
Test MGA capacity solutions with different operational weather years.

Loads MGA capacities from cache, applies them to the base network, and runs
dispatch optimization with weather from a different year.
"""

import logging
import pandas as pd
import numpy as np
import pypsa
import sys
from _helpers import (
    configure_logging,
    set_scenario_config,
    update_config_from_wildcards,
)
from solve_network import prepare_network, solve_network
from scripts._benchmark import memory_logger
from test_operations import set_weather, set_co2_price, extract_shedding_metrics

logger = logging.getLogger(__name__)


def load_mga_capacities(file_path: str) -> pd.DataFrame:
    """
    Load MGA capacities from cache CSV file.

    Expected format (long format):
    component,name,attribute,value
    Generator,DE0 0 solar,p_nom_opt,150.5
    Store,DE0 0 battery,e_nom_opt,100.0

    Parameters
    ----------
    file_path : str
        Path to MGA capacities CSV file

    Returns
    -------
    pd.DataFrame
        MGA capacities in long format
    """
    return pd.read_csv(file_path)


def apply_mga_capacities(n: pypsa.Network, capacities_df: pd.DataFrame) -> None:
    """
    Apply MGA capacity values to network components.

    Parameters
    ----------
    n : pypsa.Network
        Network to modify
    capacities_df : pd.DataFrame
        MGA capacities in long format (component, name, attribute, value)
    """
    for component in capacities_df["component"].unique():
        component_caps = capacities_df[capacities_df["component"] == component]

        for attribute in component_caps["attribute"].unique():
            attr_caps = component_caps[component_caps["attribute"] == attribute]

            # Set the attribute values
            for _, row in attr_caps.iterrows():
                name = row["name"]
                value = row["value"]

                try:
                    n.df(component).loc[name, attribute] = value
                except KeyError:
                    logger.warning(
                        f"Component {component} '{name}' not found in network. Skipping."
                    )


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "validation_mga",
            configfiles="config/mini-sector_droughts.yaml",
            opts="50seg-lv1.0",
            clusters="50",
            sector_opts="Co2L0.0+T+H+B+I+A",
            planning_horizons="2050",
            dir_hash="abc123de",  # Example hash
            operational_year="2019",
        )

    configure_logging(snakemake)
    set_scenario_config(snakemake)
    update_config_from_wildcards(snakemake.config, snakemake.wildcards)

    solve_opts = snakemake.params.solving["options"]

    # Activate load shedding
    solve_opts["load_shedding"] = True

    np.random.seed(solve_opts.get("seed", 123))

    # Load base network and weather network
    n = pypsa.Network(snakemake.input.network)
    m = pypsa.Network(snakemake.input.weather_network)

    # Load and apply MGA capacities
    logger.info(f"Loading MGA capacities from {snakemake.input.mga_capacities}")
    mga_capacities = load_mga_capacities(snakemake.input.mga_capacities)

    # Add small operational buffer to avoid numerical infeasibilities
    # when operating with different weather patterns
    buffer = 0.001  # 0.1% buffer
    logger.info(f"Adding {buffer*100:.1f}% operational buffer to all MGA capacities")
    mga_capacities['value'] *= (1 + buffer)

    apply_mga_capacities(n, mga_capacities)
    logger.info(f"Applied MGA capacities for direction {snakemake.wildcards.dir_hash}")

    planning_horizons = snakemake.wildcards.get("planning_horizons", "")

    try:
        # Apply weather and fix capacities
        set_weather(n, m)
        n.optimize.fix_optimal_capacities()

        # Prepare network
        prepare_network(
            n,
            solve_opts=solve_opts,
            foresight=snakemake.params.foresight,
            planning_horizons=planning_horizons,
            co2_sequestration_potential=snakemake.params["co2_sequestration_potential"],
            limit_max_growth=snakemake.params.get("sector", {}).get("limit_max_growth"),
        )

        # Apply CO2 price if configured
        logging_frequency = snakemake.config.get("solving", {}).get(
            "mem_logging_frequency", 30
        )
        if snakemake.config["run"]["stress_tests"].get("mode", "") == "co2-price":
            logger.info("Setting CO2 price based on previous optimization.")
            set_co2_price(n)

        # Solve network
        with memory_logger(
            filename=getattr(snakemake.log, "memory", None), interval=logging_frequency
        ) as mem:
            solve_network(
                n,
                config=snakemake.config,
                params=snakemake.params,
                solving=snakemake.params.solving,
                log_fn=snakemake.log.solver,
            )

        logger.info(f"Maximum memory usage: {mem.mem_usage}")

        # Extract shedding metrics
        load_shedding, heat_shedding = extract_shedding_metrics(n)

        # Export shedding results (no full network to save disk space)
        load_shedding.round(3).to_csv(snakemake.output.load_shedding)
        heat_shedding.round(3).to_csv(snakemake.output.heat_shedding)

        logger.info(f"MGA validation complete for direction {snakemake.wildcards.dir_hash}, year {snakemake.wildcards.operational_year}")

    except Exception as e:
        logger.error(f"Error in MGA validation: {e}")
        sys.exit(1)

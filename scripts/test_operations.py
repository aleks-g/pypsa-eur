# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2023-2025 Aleksander Grochowicz & Koen van Greevenbroek
#
# SPDX-License-Identifier: MIT
"""
Solves linear optimal dispatch in hourly resolution using the capacities of
previous capacity expansion in rule `solve_network`.
"""

import logging

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

logger = logging.getLogger(__name__)


def set_weather(
    n: pypsa.Network,
    n_weather: pypsa.Network,
) -> None:
    # TODO: this should work for the electricity only model too, where n.links_t.efficiency is just empty.
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

        n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
        n.export_to_netcdf(snakemake.output.network)

        # Save the load shedding in a dataframe.
        load_shedding = n.generators_t.p.filter(like="load shedding", axis="columns")

        # Remove "battery load" and "H2 load".
        load_shedding = load_shedding.loc[
            :, ~load_shedding.columns.str.contains("battery|H2")
        ].round(0)

        # Save heat shedding in a dataframe.
        heat_shedding = n.generators_t.p.filter(like="heat shedding", axis="columns")

        # Export the results.
        load_shedding.round(3).to_csv(snakemake.output.load_shedding)
        heat_shedding.round(3).to_csv(snakemake.output.heat_shedding)
    except Exception as e:
        logger.error(f"Error setting weather: {e}")
        sys.exit(1)

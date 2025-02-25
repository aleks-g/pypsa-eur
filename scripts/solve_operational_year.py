# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2023-2024 Aleksander Grochowicz & Koen van Greevenbroek
#
# SPDX-License-Identifier: MIT
"""
Solves linear optimal dispatch in hourly resolution using the capacities of
previous capacity expansion in rule :mod:`solve_network`.
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

logger = logging.getLogger(__name__)


def set_weather(
    n: pypsa.Network, n_weather: pypsa.Network,
) -> None:
    for c, attr in [
        ("Generator", "p_max_pu"),
        ("StorageUnit", "p_max_pu"),
        ("StorageUnit", "inflow"),
        ("Load", "p_set"),
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
            "solve_operational_years",
            configfiles="test/config.electricity.yaml",
            opts="",
            clusters="5",
            ll="v1.5",
            sector_opts="",
            planning_horizons="",
        )

    configure_logging(snakemake)
    set_scenario_config(snakemake)
    update_config_from_wildcards(snakemake.config, snakemake.wildcards)

    solve_opts = snakemake.params.options

    # Activate load shedding
    solve_opts["load_shedding"] = True

    np.random.seed(solve_opts.get("seed", 123))

    n = pypsa.Network(snakemake.input.network)
    m = pypsa.Network(snakemake.input.weather_network)
    try:
        set_weather(n, m)
        n.optimize.fix_optimal_capacities()
        n = prepare_network(n, solve_opts, config=snakemake.config)
        n = solve_network(
            n,
            config=snakemake.config,
            params=snakemake.params,
            solving=snakemake.params.solving,
            log_fn=snakemake.log.solver,
        )

        n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
        n.export_to_netcdf(snakemake.output.network)

        # Save the load shedding in a dataframe.
        load_shedding = n.generators_t.p.filter(like="load", axis="columns")

        # Remove "battery load" and "H2 load".
        load_shedding = load_shedding.loc[:,~load_shedding.columns.str.contains("battery|H2")].round(0)

        # Export the results.
        load_shedding.to_csv(snakemake.output.load_shedding)
    except Exception as e:
        logger.error(f"Error setting weather: {e}")
        sys.exit(1)

    

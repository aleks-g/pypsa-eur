# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT


rule solve_sector_network:
    params:
        solving=config_provider("solving"),
        foresight=config_provider("foresight"),
        co2_sequestration_potential=config_provider(
            "sector", "co2_sequestration_potential", default=200
        ),
        custom_extra_functionality=input_custom_extra_functionality,
    input:
        network=resources(
            "networks/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.nc"
        ),
    output:
        network=RESULTS
        + "networks/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.nc",
        config=RESULTS
        + "configs/config.base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.yaml",
    shadow:
        shadow_config
    log:
        solver=RESULTS
        + "logs/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_solver.log",
        memory=RESULTS
        + "logs/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_memory.log",
        python=RESULTS
        + "logs/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_python.log",
    threads: solver_threads
    resources:
        mem_mb=config_provider("solving", "mem_mb"),
        runtime=config_provider("solving", "runtime", default="6h"),
    benchmark:
        (
            RESULTS
            + "benchmarks/solve_sector_network/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}"
        )
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/solve_network.py"

rule test_operations:
    params:
        solving=config_provider("solving"),
        foresight=config_provider("foresight"),
        co2_sequestration_potential=config_provider(
            "sector", "co2_sequestration_potential", default=200
        ),
        custom_extra_functionality=input_custom_extra_functionality,
    input:
        network=RESULTS
        + "networks/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.nc",
        weather_network = "resources/" + config["run"]["prefix"] + "/{operational_year}/networks/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.nc",
    output:
        network=RESULTS
        + "networks/{operational_year}_base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.nc",
        load_shedding= RESULTS + "validation/{operational_year}_base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_load_shedding.csv",
        heat_shedding= RESULTS + "validation/{operational_year}_base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_heat_shedding.csv",
    shadow:
        shadow_config
    log:
        solver=RESULTS
        + "logs/{operational_year}_base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_solver.log",
        memory=RESULTS
        + "logs/{operational_year}_base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_memory.log",
        python=RESULTS
        + "logs/{operational_year}_base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_python.log",
    threads: solver_threads
    resources:
        mem_mb=config_provider("solving", "mem_mb"),
        runtime=config_provider("solving", "runtime", default="6h"),
    benchmark:
        (
            RESULTS
            + "benchmarks/test_operations/{operational_year}_base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}"
        )
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/test_operations.py"

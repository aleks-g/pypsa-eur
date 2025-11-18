# SPDX-FileCopyrightText: Aleksander Grochowicz 2025
#
# SPDX-License-Identifier: MIT


rule compute_near_opt:
    params:
        solving=config_provider("solving"),
        foresight=config_provider("foresight"),
        co2_sequestration_potential=config_provider(
            "sector", "co2_sequestration_potential", default=200
        ),
        custom_extra_functionality=input_custom_extra_functionality,
        approx=config_provider("near-opt", "approx"),
        projection=config_provider("near-opt", "projection"),
        slack=config_provider("near-opt", "slack"),
        cache_dir=RESULTS + "mga_cache/",
        cache_val=config_provider("near-opt", "cache_val", default=None),
    input:
        network=RESULTS + "networks/base_s_{clusters}_elec_{opts}.nc",
    output:
        near_opt_solutions=RESULTS + "near_opt/base_s_{clusters}_elec_{opts}_{network_hash}.csv",
    log:
        solver=normpath(
            RESULTS + "logs/mga/compute_near_opt/base_s_{clusters}_elec_{opts}_{network_hash}_solver.log"
        ),
        python=RESULTS + "logs/mga/compute_near_opt/base_s_{clusters}_elec_{opts}_{network_hash}_python.log",
    benchmark:
        (RESULTS + "benchmarks/mga/compute_near_opt/base_s_{clusters}_elec_{opts}_{network_hash}")
    threads: solver_threads * config_provider("near-opt", "approx", "max_parallel")
    resources:
        mem_mb=lambda wildcards: memory * config_provider("near-opt", "approx", "max_parallel"),
        runtime=lambda wildcards: (
            config_provider("solving", "runtime", default="6h")
            * (
                config_provider("near-opt", "iterations") // config_provider("near-opt", "approx", "max_parallel")
                + (2 * len(config_provider("near-opt", "projection")) if config_provider("near-opt", "minmax") else 0)
            )
        ),
    shadow:
        shadow_config
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/mga/compute_near_opt.py"

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


rule validation_mga:
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
        weather_network="resources/" + config["run"]["prefix"] + "/{operational_year}/networks/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.nc",
        mga_capacities=config_provider("near-opt", "cache_dir") + "/caps_{network_hash}_{dir_hash}.csv",
    output:
        load_shedding=RESULTS + "validation/mga_{network_hash}_{dir_hash}_{operational_year}_base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_load_shedding.csv",
        heat_shedding=RESULTS + "validation/mga_{network_hash}_{dir_hash}_{operational_year}_base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_heat_shedding.csv",
    shadow:
        shadow_config
    log:
        solver=RESULTS
        + "logs/mga/validation_mga_{network_hash}_{dir_hash}_{operational_year}_base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_solver.log",
        memory=RESULTS
        + "logs/mga/validation_mga_{network_hash}_{dir_hash}_{operational_year}_base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_memory.log",
        python=RESULTS
        + "logs/mga/validation_mga_{network_hash}_{dir_hash}_{operational_year}_base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_python.log",
    threads: solver_threads
    resources:
        mem_mb=config_provider("solving", "mem_mb"),
        runtime=config_provider("solving", "runtime", default="6h"),
    benchmark:
        (
            RESULTS
            + "benchmarks/mga/validation_mga_{network_hash}_{dir_hash}_{operational_year}_base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}"
        )
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/test_mga_operations.py"

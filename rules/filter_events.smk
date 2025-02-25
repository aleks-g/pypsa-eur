# SPDX-FileCopyrightText: : 2024 Aleksander Grochowicz
#
# SPDX-License-Identifier: MIT

import yaml

def load_operational_years(file_path):
    with open(file_path, 'r') as file:
        return list(yaml.safe_load(file).keys())

def input_difficult_periods(wildcards):
    return expand(
        RESULTS +
        "networks/base_s_{clusters}_elec_l{ll}_{opts}.nc",
        clusters=wildcards.clusters,
        ll=wildcards.ll,
        opts=wildcards.opts,
        run = config["run"]["name"]
    )

# For now, only for the electricity sector.
rule all_difficult_periods:
    input:
        expand(
            "results/periods/sde_{scen_name}_{clusters}_elec_l{ll}_{opts}.csv",
            **config["scenario"],
            run=config["run"]["name"],
            scen_name=config["difficult_periods"]["scen_name"]
        ),



rule difficult_periods:
    input:
        input_difficult_periods,
    output:
        "results/periods/sde_{scen_name}_{clusters}_elec_l{ll}_{opts}.csv"
    log:
        python="results/logs/difficult_periods/sde_{scen_name}_{clusters}_elec_l{ll}_{opts}_python.log", 
    benchmark:
        "results/benchmarks/difficult_periods/sde_{scen_name}_{clusters}_elec_l{ll}_{opts}"
    threads: 16
    resources:
        mem_mb=20000,
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/difficult_periods.py" 











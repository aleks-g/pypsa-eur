# SPDX-FileCopyrightText: : 2024 Aleksander Grochowicz
#
# SPDX-License-Identifier: MIT

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
        expand(
            RESULTS
            + "networks/base_s_{clusters}_elec_l{ll}_{opts}.nc",
            **config["scenario"],
            run=config["run"]["name"]
        ),
    output:
        "results/periods/sde_{scen_name}_{clusters}_elec_l{ll}_{opts}.csv"
    log:
        python=RESULTS
        + "logs/difficult_periods/sde_{scen_name}_{clusters}_elec_l{ll}_{opts}_python.log", 
    benchmark:
        "results/benchmarks/difficult_periods/sde_{scen_name}_{clusters}_elec_l{ll}_{opts}"
    threads: 1
    resources:
        mem_mb=4000,
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/difficult_periods.py" 




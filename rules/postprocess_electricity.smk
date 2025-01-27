# SPDX-FileCopyrightText: : 2024 Aleksander Grochowicz
#
# SPDX-License-Identifier: MIT
#
# Note that this is just temporary to add postprocessing functionality without running the sector-coupled model.


rule all_statistics:
    input:
        expand(
            RESULTS + "csvs/{file}.csv",
            file=[
                "nodal_costs",
                "nodal_capacities",
                "nodal_cfs",
                "cfs",
                "costs",
                "capacities",
                "curtailment",
                "energy",
                "supply",
                "supply_energy",
                "nodal_supply_energy",
                "prices",
                "market_values",
                "price_statistics",
                "metrics",
            ],
            **config["scenario"],
            run=config["run"]["name"]
        ),
        expand(
            RESULTS + "graphs/{file}.svg",
            file=[
                "costs",
                "energy",
            ],
            **config["scenario"],
            run=config["run"]["name"]
        ),
        


rule plot_power_network_clustered_elec:
    params:
        plotting=config_provider("plotting"),
    input:
        network=resources("networks/base_s_{clusters}.nc"),
        regions_onshore=resources("regions_onshore_base_s_{clusters}.geojson"),
    output:
        map=resources("maps/power-network-s-{clusters}.pdf"),
    threads: 1
    resources:
        mem_mb=4000,
    benchmark:
        benchmarks("plot_power_network_clustered/base_s_{clusters}")
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/plot_power_network_clustered_elec.py"

# Note that this has been adapted
rule plot_power_network_elec:
    params:
        plotting=config_provider("plotting"),
    input:
        network=RESULTS + "networks/base_s_{clusters}_elec_l{ll}_{opts}.nc",
        regions=resources("regions_onshore_base_s_{clusters}.geojson"),
    output:
        map=RESULTS
        + "maps/base_s_{clusters}_elec_l{ll}_{opts}-costs.pdf",
    threads: 2
    resources:
        mem_mb=10000,
    log:
        RESULTS
        + "logs/plot_power_network/base_s_{clusters}_elec_l{ll}_{opts}.log",
    benchmark:
        (
            RESULTS
            + "benchmarks/plot_power_network/base_s_{clusters}_elec_l{ll}_{opts}"
        )
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/plot_power_network_elec.py"

rule make_summary_elec:
    params:
        foresight=config_provider("foresight"),
        costs=config_provider("costs"),
        snapshots=config_provider("snapshots"),
        drop_leap_day=config_provider("enable", "drop_leap_day"),
        scenario=config_provider("scenario"),
        RDIR=RDIR,
    input:
        networks=expand(
            RESULTS
            + "networks/base_s_{clusters}_elec_l{ll}_{opts}.nc",
            **config["scenario"],
            allow_missing=True,
        ),
        costs=lambda w: (
            resources("costs_{}.csv".format(config_provider("costs", "year")(w)))
            if config_provider("foresight")(w) == "overnight"
            else resources(
                "costs_{}.csv".format(
                    config_provider("scenario", "planning_horizons", 0)(w)
                )
            )
        ),
        ac_plot=expand(
            resources("maps/power-network-s-{clusters}.pdf"),
            **config["scenario"],
            allow_missing=True,
        ),
        costs_plot=expand(
            RESULTS
            + "maps/base_s_{clusters}_elec_l{ll}_{opts}-costs.pdf",
            **config["scenario"],
            allow_missing=True,
        ),
    output:
        nodal_costs=RESULTS + "csvs/nodal_costs.csv",
        nodal_capacities=RESULTS + "csvs/nodal_capacities.csv",
        nodal_cfs=RESULTS + "csvs/nodal_cfs.csv",
        cfs=RESULTS + "csvs/cfs.csv",
        costs=RESULTS + "csvs/costs.csv",
        capacities=RESULTS + "csvs/capacities.csv",
        curtailment=RESULTS + "csvs/curtailment.csv",
        energy=RESULTS + "csvs/energy.csv",
        supply=RESULTS + "csvs/supply.csv",
        supply_energy=RESULTS + "csvs/supply_energy.csv",
        nodal_supply_energy=RESULTS + "csvs/nodal_supply_energy.csv",
        prices=RESULTS + "csvs/prices.csv",
        market_values=RESULTS + "csvs/market_values.csv",
        price_statistics=RESULTS + "csvs/price_statistics.csv",
        metrics=RESULTS + "csvs/metrics.csv",
    threads: 2
    resources:
        mem_mb=10000,
    log:
        RESULTS + "logs/make_summary.log",
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/make_summary_elec.py"


rule plot_summary_elec:
    params:
        countries=config_provider("countries"),
        planning_horizons=config_provider("scenario", "planning_horizons"),
        emissions_scope=config_provider("energy", "emissions"),
        plotting=config_provider("plotting"),
        foresight=config_provider("foresight"),
        co2_budget=config_provider("co2_budget"),
        RDIR=RDIR,
    input:
        costs=RESULTS + "csvs/costs.csv",
        energy=RESULTS + "csvs/energy.csv",
        balances=RESULTS + "csvs/supply_energy.csv",
        eurostat="data/eurostat/Balances-April2023",
        co2="data/bundle/eea/UNFCCC_v23.csv",
    output:
        costs=RESULTS + "graphs/costs.svg",
        energy=RESULTS + "graphs/energy.svg",
    threads: 2
    resources:
        mem_mb=10000,
    log:
        RESULTS + "logs/plot_summary.log",
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/plot_summary_elec.py"



STATISTICS_BARPLOTS = [
    "capacity_factor",
    "installed_capacity",
    "optimal_capacity",
    "optimal_store",
    "capital_expenditure",
    "operational_expenditure",
    "curtailment",
    "supply",
    "withdrawal",
    "market_value",
]

rule plot_base_statistics_elec:
    params:
        plotting=config_provider("plotting"),
        barplots=STATISTICS_BARPLOTS,
    input:
        network=RESULTS + "networks/base_s_{clusters}_elec_l{ll}_{opts}.nc",
    output:
        **{
            f"{plot}_bar": RESULTS
            + f"figures/statistics_{plot}_bar_base_s_{{clusters}}_elec_l{{ll}}_{{opts}}.pdf"
            for plot in STATISTICS_BARPLOTS
        },
        barplots_touch=RESULTS
        + "figures/.statistics_plots_base_s_{clusters}_elec_l{ll}_{opts}",
    script:
        "../scripts/plot_statistics_elec.py"

rule compute_means:
    input:
        expand(
            RESULTS
            + "networks/base_s_{clusters}_elec_l{ll}_{opts}.nc",
            **config["scenario"],
            run=config["run"]["name"]
        ),
    output:
        wind="results/means/wind_{scen_name}_{clusters}_elec_l{ll}_{opts}.csv",
        solar="results/means/solar_{scen_name}_{clusters}_elec_l{ll}_{opts}.csv",
        load="results/means/load_{scen_name}_{clusters}_elec_l{ll}_{opts}.csv",
    log:
        "results/periods/means_{scen_name}_{clusters}_elec_l{ll}_{opts}.csv"
    conda:
        "../envs/environment.yaml"
    resources:
        mem_mb=5000,
    threads: 1
    script:
        "../scripts/compute_means.py"
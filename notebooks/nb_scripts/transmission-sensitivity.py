# %%
# Initiation

import pypsa 
import datetime as dt 
import yaml

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.path import Path
import seaborn as sns

mpl.rcParams["figure.dpi"] = 150

import pandas as pd
import numpy as np

import xarray as xr
from scipy.stats import gaussian_kde
from scipy.stats import wasserstein_distance

from matplotlib.ticker import (AutoMinorLocator, MultipleLocator, FormatStrFormatter,AutoMinorLocator)
import matplotlib.dates as mdates
from matplotlib.path import Path
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors

from typing import NamedTuple, Optional

from _notebook_utilities import *
from _generate_data_for_analysis import *

import logging

# Suppress warnings and info messages from 'pypsa.io'
logging.getLogger("pypsa.io").setLevel(logging.ERROR)

cm = 1 / 2.54  # centimeters in inches



# %%
regenerate_data = False
overwrite_data = False
config_name = "stressful-weather"

# In order to regenerate data, run `generate_data_for_analysis.py`.

# %%
config, scenario_def, years, opt_networks = load_opt_networks(config_name, load_networks=regenerate_data)


# %%
periods = load_periods(config)
folder = f"./processing_data/{config_name}"
total_costs_df = pd.read_csv(f"{folder}/total_costs.csv", index_col=[0,1])
total_costs = {}
for year in years:
    df = total_costs_df.loc[year]
    df.index = pd.to_datetime(df.index)
    total_costs[year] = df["0"]

# %%
def load_sensitivity_periods(config: dict):
    '''Load the system-defining events based on the configuration.'''
    scen_name = config["difficult_periods"]["scen_name"]
    clusters = config["scenario"]["clusters"]
    ll = config["scenario"]["ll"]
    opts = config["scenario"]["opts"]
    periods_name = f"sde_{scen_name}_{clusters[0]}_elec_l{ll[0]}_{opts[0]}"

    periods = pd.read_csv(f"../results/periods/{periods_name}.csv", index_col=0, parse_dates=["start", "end", "peak_hour"])

    for col in ["start", "end", "peak_hour"]:
        periods[col] = periods[col].dt.tz_localize(None)
    return periods

# %%
sens_config_name = "stressful-weather-sensitivities"
cost_thresholds = [30,50, 100]
dict_names = [f"new_store_1941-2021_{c}bn_12-336h" for c in cost_thresholds]

sens_config, sens_scenario_def, years, sens_opt_networks = load_opt_networks(sens_config_name, load_networks=regenerate_data)


# %%
def load_sens_periods(sens_config, cost_thresholds):
    '''Load the system-defining events based on the configuration.'''
    file_names = {cost_threshold: {} for cost_threshold in cost_thresholds}
    sens_periods = {cost_threshold: {} for cost_threshold in cost_thresholds}
    for cost_threshold in cost_thresholds:
        dict_name = f"new_store_1941-2021_{cost_threshold}bn_12-336h"
        clusters = sens_config["scenario"]["clusters"]
        lls = sens_config["scenario"]["ll"]
        optss = sens_config["scenario"]["opts"]

        # Take the product of the lists of clusters, lls and optss
        for cluster in clusters:
            for ll in lls:
                for opts in optss:
                    periods_name = f"sde_{dict_name}_{cluster}_elec_l{ll}_{opts}"
                    file_names[cost_threshold][(cluster, ll, opts)] = periods_name
                    sens_periods[cost_threshold][(cluster, ll, opts)] = pd.read_csv(f"../results/periods/{periods_name}.csv", index_col=0, parse_dates=["start", "end", "peak_hour"])

    return sens_periods
    

# %%
sens_periods = load_sens_periods(sens_config, cost_thresholds)

# %%
for j, scen in enumerate(sens_periods[30].keys()):
    print(j, scen)

# %%
if regenerate_data:
    config_c1, scenario_def_c1, years, opt_networks_c1 = load_opt_networks("stressful-weather-sensitivities", config_str = "base_s_90_elec_lc1.0_Co2L0.0", load_networks=True)
    trans_costs, _, _ = compute_all_duals(opt_networks_c1) 
    if overwrite_data:
        pd.concat(trans_costs).to_csv(f"sensitivity_analysis/c1_costs.csv")
else:
    trans_costs_df = pd.read_csv(f"sensitivity_analysis/c1_costs.csv", index_col=[0,1])
    trans_costs = {}
    for year in years:
        df = trans_costs_df.loc[year]
        df.index = pd.to_datetime(df.index)
        trans_costs[year] = df["0"]



# %%
# Standard vs c1.0
plot_duals(
        periods, 
        years, 
        total_costs, 
        trans_costs, 
        mpl.colors.LogNorm(vmin=0.1, vmax=10000, clip=True),
        mpl.colors.LogNorm(vmin=0.1, vmax=10000, clip=True), 
        "Oranges",
        "Greens",
        "Hourly electricity cost (million EUR)",
        "Hourly electricity cost (million EUR)",
        [1, 10, 100, 1000, 10000],
        [1, 10, 100, 1000, 10000],
        save_fig=False,
        path_str = None,
        alt_periods = sens_periods[100][(90, 'c1.0', 'Co2L0.0')]
    )

# %% [markdown]
# # Comparison transmission

# %%
config_c1, scenario_def_c1, years, opt_networks_c1 = load_opt_networks("stressful-weather-sensitivities", config_str = "base_s_90_elec_lc1.0_Co2L0.0", load_networks=regenerate_data)

# %%
opt_obj = pd.read_csv("./processing_data/stressful-weather/opt_objs.csv", index_col=0)
if regenerate_data:
    opt_obj_c1, _ = optimal_costs(opt_networks_c1)
    if overwrite_data:
        opt_obj_c1.to_csv(f"sensitivity_analysis/c1_opt_objs.csv")
else:
    opt_obj_c1 = pd.read_csv(f"sensitivity_analysis/c1_opt_objs.csv", index_col=0)
    

# %%
if regenerate_data:
    net_load_c1 = compute_net_load(opt_networks_c1, opt_networks_c1[1941])
    carrier_tech = ["biomass", "nuclear", "offwind", "solar", "onwind", "ror"]
    links_tech = ["H2 fuel cell", "battery discharger"]
    su_tech = ["PHS", "hydro"]

    peak_gen = pd.DataFrame(columns = carrier_tech + links_tech + su_tech + ["net load"], index = periods.peak_hour).astype(float)

    for period in periods.index:
        peak_hour = periods.loc[period, "peak_hour"]
        net_year = get_net_year(peak_hour)
        n = opt_networks_c1[net_year]
        
        for tech in carrier_tech:
            if tech == "offwind":
                c_id = n.generators.loc[n.generators.carrier.str.contains("offwind")].index
            elif tech == "solar":
                c_id = n.generators.loc[n.generators.carrier.str.contains("solar")].index
            else:
                c_id = n.generators.loc[n.generators.carrier == tech].index
            peak_gen.loc[peak_hour,tech] = n.generators_t.p.loc[peak_hour, c_id].sum()
        for tech in links_tech:
            l_id = n.links.loc[n.links.carrier == tech].index
            peak_gen.loc[peak_hour,tech] = n.links_t.p1.loc[peak_hour, l_id].abs().sum()
        for tech in su_tech:
            su_id = n.storage_units.loc[n.storage_units.carrier == tech].index
            peak_gen.loc[peak_hour,tech] = n.storage_units_t.p.loc[peak_hour, su_id].sum()
        peak_gen.loc[peak_hour,"net load"] = net_load_c1.loc[peak_hour]

    peak_gen /= 1e3 # in GW
    peak_gen_c1 = peak_gen.round(1)

    if overwrite_data:
        net_load_c1.to_csv("sensitivity_analysis/net_load_c1.csv")
        peak_gen_c1.to_csv("sensitivity_analysis/peak_gen_c1.csv")
else:
    net_load_c1 = pd.read_csv("sensitivity_analysis/net_load_c1.csv", index_col=0, parse_dates=True)
    peak_gen_c1 = pd.read_csv("sensitivity_analysis/peak_gen_c1.csv", index_col=0, parse_dates=True)
    peak_gen = pd.read_csv("processing_data/stressful-weather/peak_gen.csv", index_col=0, parse_dates=True)
    net_load = pd.read_csv("processing_data/stressful-weather/net_load.csv", index_col=0, parse_dates=True)

# %%
# Plot a stacked bar plot of the peak generation per technology for each event.
fig, ax = plt.subplots(1,1, figsize = (18*cm, 9*cm))
reindexed_peak_gen = peak_gen_c1.reset_index()[["biomass", "nuclear", "ror", "battery discharger", "hydro", "PHS", "H2 fuel cell", "solar", "offwind", "onwind"]]

colours["H2 fuel cell"] = colours["fuel_cells"]
colours["battery discharger"] = colours["battery"]
colours["PHS"] = colours["phs"]
colours["offwind"] = "#6895dd"
colours["onwind"] = "#235ebc"
colours["solar"] = "#f9d002"
colours["OCGT"] = '#e0986c'
colours["CCGT"] = '#a85522'

reindexed_peak_gen.plot.bar(
    stacked=True,
    ax=ax,
    width = 0.8,
    color=[colours[tech] for tech in reindexed_peak_gen.columns],
);
# Draw a red line for the net load.
ax.plot(reindexed_peak_gen.index, peak_gen_c1["net load"], color="black", linewidth=0, marker = "_", label="Net load")
ax.legend(bbox_to_anchor=(0, -0.1), loc=2, ncol=4, borderaxespad=0.0, frameon=False, labelspacing=0.75);
ax.set_title("Generation in peak hour per SDE");


# %%
# Plot a stacked bar plot of the peak generation per technology for each event.
fig, ax = plt.subplots(1,1, figsize = (18*cm, 9*cm))
reindexed_peak_gen = (peak_gen_c1 - peak_gen).reset_index()[["biomass", "nuclear", "ror", "battery discharger", "hydro", "PHS", "H2 fuel cell", "solar", "offwind", "onwind"]]

colours["H2 fuel cell"] = colours["fuel_cells"]
colours["battery discharger"] = colours["battery"]
colours["PHS"] = colours["phs"]
colours["offwind"] = "#6895dd"
colours["onwind"] = "#235ebc"
colours["solar"] = "#f9d002"
colours["OCGT"] = '#e0986c'
colours["CCGT"] = '#a85522'

reindexed_peak_gen.plot.bar(
    stacked=True,
    ax=ax,
    width = 0.8,
    color=[colours[tech] for tech in reindexed_peak_gen.columns],
);
# Draw a red line for the net load.
ax.plot(reindexed_peak_gen.index, peak_gen_c1["net load"] - peak_gen["net load"], color="black", linewidth=0, marker = "_", label="Net load")
ax.legend(bbox_to_anchor=(0, -0.1), loc=2, ncol=4, borderaxespad=0.0, frameon=False, labelspacing=0.75);
ax.set_title("Different generation without transmission expansion in peak hour per SDE");


# %%
opt_obj

# %%
((opt_obj_c1 - opt_obj)/1e9).astype(float).describe()

# %%
((opt_obj_c1 - opt_obj)/1e9).astype(float)["Onshore wind"].plot()

# %%
((opt_obj_c1 - opt_obj)/1e9).astype(float)["Onshore wind"].sort_values()

# %%
((opt_obj_c1 - opt_obj)/1e9).astype(float)[["Onshore wind", "Offshore wind"]].sum(axis="columns").plot()
plt.hlines(xmin=0, xmax=80, y=((opt_obj_c1 - opt_obj)/1e9).astype(float)[["Onshore wind","Offshore wind"]].sum(axis="columns").mean(), color="red", ls="--")
# plt.xlim(0,80)

# %%
((opt_obj_c1 - opt_obj)/1e9).astype(float)["Onshore wind"].plot()
plt.hlines(xmin=0, xmax=80, y=((opt_obj_c1 - opt_obj)/1e9).astype(float)["Onshore wind"].mean(), color="red", ls="--")
plt.xlim(0,80);

# %%
((opt_obj_c1 - opt_obj)/1e9).astype(float)["Offshore wind"].plot()
plt.hlines(xmin=0, xmax=80, y=((opt_obj_c1 - opt_obj)/1e9).astype(float)["Offshore wind"].mean(), color="red", ls="--")
plt.xlim(0,80);

# %%
((opt_obj_c1.sum(axis="columns") - opt_obj.sum(axis="columns"))/1e9).sort_values()

# %%
if regenerate_data:
    storage_caps_orig = pd.DataFrame(index = years, columns = ["H2", "H2 fuel cell", "H2 electrolysis", "battery", "battery discharger"])

    for year in years:
        for col in storage_caps_orig.columns:
            if col in ["H2", "battery"]:
                storage_caps_orig.loc[year, col] = opt_networks[year].stores[opt_networks[year].stores.carrier == col].sum().e_nom_opt/1e3 # GWh
            else:
                storage_caps_orig.loc[year, col] = (opt_networks[year].links[opt_networks[year].links.carrier == col].sum().p_nom_opt)/1e3 # GW
    storage_caps_orig = storage_caps_orig.astype(float).round(1)
    if overwrite_data:
        storage_caps_orig.to_csv(f"sensitivity_analysis/storage_caps_orig.csv")
else:
    storage_caps_orig = pd.read_csv(f"sensitivity_analysis/storage_caps_orig.csv", index_col=0)


# %%
storage_caps_orig.describe().round(1)

# %%
if regenerate_data:
    storage_caps = pd.DataFrame(index = years, columns = ["H2", "H2 fuel cell", "H2 electrolysis", "battery", "battery discharger"])

    for year in years:
        for col in storage_caps.columns:
            if col in ["H2", "battery"]:
                storage_caps.loc[year, col] = opt_networks_c1[year].stores[opt_networks_c1[year].stores.carrier == col].sum().e_nom_opt/1e3 # GWh
            else:
                storage_caps.loc[year, col] = (opt_networks_c1[year].links[opt_networks_c1[year].links.carrier == col].sum().p_nom_opt)/1e3 # GW
    storage_caps = storage_caps.astype(float).round(1)
    if overwrite_data:
        storage_caps.to_csv(f"sensitivity_analysis/c1_storage_caps.csv")
else:
    storage_caps = pd.read_csv(f"sensitivity_analysis/c1_storage_caps.csv", index_col=0)


# %%
(storage_caps - storage_caps_orig).round(2)

# %%
# New SDEs: required more hydrogen capacity (a little more battery), much less DC investment
# These are years with generally less onshore wind, but more hydrogen investment, slightly less battery (similar solar). Similar average costs, slightly lower mean, higher median.
# In these years there was significant additional investment in fuel cell capacities as well as H2 storage. A bit more battery discharger too, but not as much in relation.
display((((opt_obj_c1 - opt_obj)/1e9).loc[["56/57", "57/58","71/72","87/88","90/91","06/07","11/12","12/13","13/14"]]).astype(float).round(2))
(((opt_obj_c1)/1e9).loc[["56/57", "57/58","71/72","87/88","90/91","06/07","11/12","12/13","13/14"]]).astype(float).sum(axis="columns").describe().round(2)
# Most of these events were captured through our weather investigation (all but 56/57 and 57/58 and 13/14). None of them were particularly bad, but 12/13 had a strong wind anomaly, and 87/88 and 06/07 were more severe than the remaining ones.

# %%
storage_caps.loc[[1956, 1957, 1971, 1987, 1990, 2006, 2011, 2012, 2013]].describe().round(1)


# %%
storage_caps_orig.loc[[1956, 1957, 1971, 1987, 1990, 2006, 2011, 2012, 2013]].describe().round(1)

# %%
(storage_caps - storage_caps_orig).loc[[1956, 1957, 1971, 1987, 1990, 2006, 2011, 2012, 2013]].describe().round(1)

# %%
# No more SDEs: most events that were eliminated were already long; possibly the prices were stretched out more, so we didn't spike enough. The lack of transmission made them less impacted by the wind anomalies together with a higher reliance on storage. These are however generally years with more installed onshore wind, but much less fuel cells, slightly more batteries.
display(((opt_obj_c1 - opt_obj)/1e9).loc[[ "46/47","66/67","80/81","81/82","82/83","10/11"]].astype(float).round(2))
((opt_obj_c1)/1e9).loc[[ "46/47","66/67","80/81","81/82","82/83","10/11"]].astype(float).sum(axis="columns").describe().round(2)
# 46/47 was a long event with low FC capacities and not super extreme (difficult year) - Cluster 1 
# 66/67 was long but had some extreme peaks, and several spikes - Cluster 0
# 80/81 was a long event in a good year with low FC capacities, and not very extreme - Cluster 0
# 81/82 was reasonably long, but not very extreme at all, also easy year - Cluster 0
# 82/83 was long, not very extreme, but large FC capacities (rarely used) - Cluster 0
# 10/11 was short, reasonably difficult, mid FC capacities - Cluster 3

# %%
storage_caps.loc[[1946, 1966, 1980, 1981, 1982, 2010]].describe().round(1)

# %%
storage_caps_orig.loc[[1946, 1966, 1980, 1981, 1982, 2010]].describe().round(1)

# %%
(storage_caps - storage_caps_orig).loc[[1946, 1966, 1980, 1981, 1982, 2010]].describe().round(1)

# %% [markdown]
# # Load duration curves

# %%
# Compute load duration curves of fuel cells.
if regenerate_data:
    fc_i_c1 = opt_networks_c1[1941].links[opt_networks_c1[1941].links.carrier == "H2 fuel cell"].index
    gen_fc_df_c1 = pd.DataFrame(index = range(8760))
    for year, n in opt_networks_c1.items():
        gen_fc_df_c1[year] = -n.links_t.p1[fc_i_c1].sum(axis=1).sort_values(ascending=True).values
    gen_fc_df_c1 = (gen_fc_df_c1/1e3).round(1)
    if overwrite_data:
        gen_fc_df_c1.to_csv("sensitivity_analysis/gen_fc_df_c1.csv")
else:
    gen_fc_df_c1 = pd.read_csv("sensitivity_analysis/gen_fc_df_c1.csv", index_col=0)


# %%
gen_fc_df = pd.read_csv(f"processing_data/{config_name}/gen_fc_df.csv", index_col=0)

# %%
fig, ax = plt.subplots(1, 1, figsize=(18 * cm, 9 * cm))

# Plot individual load duration curves
for col in gen_fc_df_c1.columns:
    sns.lineplot(data=gen_fc_df_c1[col], ax=ax, color="grey", alpha=0.1, linewidth=0.5)


# Plot mean load duration curve
sns.lineplot(data=gen_fc_df_c1.mean(axis="columns"), ax=ax, color="red", linewidth=1, label="Mean")

# Plot fill between 10th and 90th percentiles
ax.fill_between(
    gen_fc_df_c1.index,
    gen_fc_df_c1.quantile(0.1, axis=1),
    gen_fc_df_c1.quantile(0.9, axis=1),
    color="blue",
    alpha=0.3,
    label="10th-90th percentile"
)

ax.set_ylabel("Fuel cell dispatch [GW]")
ax.set_xlabel("Hours")
ax.set_xlim(0, 1000)
# Add for each of the xticklabels and yticklabels the capacity factor (relative to 194 GW installed capacity) as well as the duration relative to 8760 hours.
ax.legend()
plt.show()

# %%
fig, axs = plt.subplots(1, 2, figsize=(30 * cm, 9 * cm))

for ax in axs:
    # Plot individual load duration curves
    for col in gen_fc_df_c1.columns:
        sns.lineplot(data=gen_fc_df_c1[col], ax=ax, color="grey", alpha=0.1, linewidth=0.5)

    # Plot mean load duration curve
    sns.lineplot(data=gen_fc_df_c1.mean(axis="columns"), ax=ax, color="red", linewidth=1, label="Mean")

    # Plot fill between 10th and 90th percentiles
    ax.fill_between(
        gen_fc_df_c1.index,
        gen_fc_df_c1.quantile(0.1, axis=1),
        gen_fc_df_c1.quantile(0.9, axis=1),
        color="blue",
        alpha=0.3,
        label="10th-90th percentile"
    )
    if ax == axs[1]:
        for col in gen_fc_df.columns:
            sns.lineplot(data=gen_fc_df[col], ax=ax, color="grey", alpha=0.1, linewidth=0.5)
        sns.lineplot(data=gen_fc_df.mean(axis="columns"), ax=ax, color="green", linewidth=1, label="Orig Mean")
        ax.fill_between(
            gen_fc_df.index,
            gen_fc_df.quantile(0.1, axis=1),
            gen_fc_df.quantile(0.9, axis=1),
            color="green",
            alpha=0.3,
            label="Orig 10th-90th percentile"
        )
    ax.set_xlabel("Hours")
    ax.set_xlim(0, 600)
    ax.set_ylim(0, 120)
   
    xticks = ax.get_xticks()
    xticklabels = [f"{xtick:.0f} \n {int(xtick/8760*100)}%" for xtick in xticks]
    ax.set_xticklabels(xticklabels)
    ax.set_ylabel("Fuel cell dispatch [GW]")
    ax.legend()
plt.show()

# %%
# Compute load duration curves of firm flexibility: nuclear, biomass, hydro, PHS
if regenerate_data:
    firm_gen_i = opt_networks_c1[1941].generators[opt_networks_c1[1941].generators.carrier.isin(["nuclear", "biomass", "ror"])].index
    firm_su_i = opt_networks_c1[1941].storage_units[opt_networks_c1[1941].storage_units.carrier.isin(["hydro", "PHS"])].index

    firm_gen_df = pd.DataFrame(index = range(8760))
    for year, n in opt_networks_c1.items():
        df_helper = pd.DataFrame(index = range(8760), columns=["prod"])
        df_helper["prod"] = n.generators_t.p[firm_gen_i].sum(axis=1).values
        df_helper["prod"] += n.storage_units_t.p[firm_su_i].sum(axis=1).values
        firm_gen_df[year] = df_helper.sort_values(by="prod", ascending=False).values
    firm_gen_df = (firm_gen_df/1e3).round(1)
    if overwrite_data:
        firm_gen_df.to_csv("sensitivity_analysis/c1_firm_gen_df.csv")
else:
    firm_gen_df = pd.read_csv("sensitivity_analysis/c1_firm_gen_df.csv", index_col=0)


# %%
firm_gen_df_orig = pd.read_csv(f"processing_data/{config_name}/firm_gen_df.csv", index_col=0)

# %%
# Plot firm flexibility: nuclear, biomass, ror, hydro, PHS
fig, axs = plt.subplots(1, 2, figsize=(30 * cm, 9 * cm),sharey=True)

for ax, df, label in zip(axs, [firm_gen_df_orig, firm_gen_df], ["c1.25", "c1"]):
    # Plot individual load duration curves
    for col in df.columns:
        sns.lineplot(data=df[col], ax=ax, color="grey", alpha=0.1, linewidth=0.5)
    
    # Plot mean load duration curve
    sns.lineplot(data=df.mean(axis="columns"), ax=ax, color="red", linewidth=1, label="Mean")

    # Plot fill between 10th and 90th percentiles
    ax.fill_between(
        df.index,
        df.quantile(0.1, axis=1),
        df.quantile(0.9, axis=1),
        color="blue",
        alpha=0.3,
        label="10th-90th percentile"
    )
    ax.set_ylabel("Firm generation [GW]");
    ax.set_xlabel("Hours");
    ax.set_xlim(0,8760);
    ax.set_title(f"{label}")



# %%
# Compute load duration curves of firm flexibility: nuclear, biomass, hydro, PHS
if regenerate_data:
    firm_gen_i = opt_networks_c1[1941].generators[opt_networks_c1[1941].generators.carrier.isin(["nuclear", "biomass", "ror"])].index
    firm_su_i = opt_networks_c1[1941].storage_units[opt_networks_c1[1941].storage_units.carrier.isin(["hydro", "PHS"])].index
    firm_s_i = opt_networks_c1[1941].links[opt_networks_c1[1941].links.carrier.isin(["battery discharger"])].index

    firm_daily_df = pd.DataFrame(index = range(8760))
    for year, n in opt_networks_c1.items():
        df_helper = pd.DataFrame(index = range(8760), columns=["prod"])
        df_helper["prod"] = n.generators_t.p[firm_gen_i].sum(axis=1).values
        df_helper["prod"] += n.storage_units_t.p[firm_su_i].sum(axis=1).values
        df_helper["prod"] -= n.links_t.p1[firm_s_i].sum(axis=1).values
        firm_daily_df[year] = df_helper.sort_values(by="prod", ascending=False).values
    firm_daily_df = (firm_daily_df/1e3).round(1)
    if overwrite_data:
        firm_daily_df.to_csv("sensitivity_analysis/c1_firm_daily_df.csv")
else:
    firm_daily_df = pd.read_csv("sensitivity_analysis/c1_firm_daily_df.csv", index_col=0)


# %%
firm_daily_df_orig = pd.read_csv(f"processing_data/{config_name}/firm_daily_df.csv", index_col=0)

# %%
# Plot daily flexibility: nuclear, biomass, hydro, PHS, BATTERY
fig, ax = plt.subplots(1, 1, figsize=(18 * cm, 9 * cm), sharey=True)

for color, df, label in zip(["green", "red"], [firm_daily_df_orig, firm_daily_df], ["original", "c1"]):
    # Plot individual load duration curves
    for col in df.columns:
        sns.lineplot(data=df[col], ax=ax, color=color, alpha=0.1, linewidth=0.5)

    # Plot mean load duration curve
    sns.lineplot(data=df.mean(axis="columns"), ax=ax, color=color, linewidth=1, label=f"Mean {label}")

    # Plot fill between 10th and 90th percentiles
    ax.fill_between(
        df.index,
        df.quantile(0.1, axis=1),
        df.quantile(0.9, axis=1),
        color=color,
        alpha=0.3,
        label=f"10th-90th percentile {label}"
    )
ax.set_ylabel("Daily balancing generation [GW]");
ax.set_xlabel("Hours");
ax.set_xlim(0,1000);
ax.legend();
ax.set_ylim(bottom=0);




# %%
# Compute load duration curve of all flexibility.
if regenerate_data:
    firm_gen_i = opt_networks_c1[1941].generators[opt_networks_c1[1941].generators.carrier.isin(["nuclear", "biomass", "ror"])].index
    firm_su_i = opt_networks_c1[1941].storage_units[opt_networks_c1[1941].storage_units.carrier.isin(["hydro", "PHS"])].index
    firm_s_i = opt_networks_c1[1941].links[opt_networks_c1[1941].links.carrier.isin(["battery discharger", "H2 fuel cell"])].index

    firm_flex_df = pd.DataFrame(index = range(8760))
    for year, n in opt_networks_c1.items():
        df_helper = pd.DataFrame(index = range(8760), columns=["prod"])
        df_helper["prod"] = n.generators_t.p[firm_gen_i].sum(axis=1).values
        df_helper["prod"] += n.storage_units_t.p[firm_su_i].sum(axis=1).values
        df_helper["prod"] -= n.links_t.p1[firm_s_i].sum(axis=1).values
        firm_flex_df[year] = df_helper.sort_values(by="prod", ascending=False).values
    firm_flex_df = (firm_flex_df/1e3).round(1)
    if overwrite_data:
        firm_flex_df.to_csv("sensitivity_analysis/c1_firm_flex_df.csv")
else:
    firm_flex_df = pd.read_csv("sensitivity_analysis/c1_firm_flex_df.csv", index_col=0)


# %%
firm_flex_df_orig = pd.read_csv(f"processing_data/{config_name}/firm_flex_df.csv", index_col=0)

# %%
# Plot all flexibility
fig, axs = plt.subplots(1, 2, figsize=(30 * cm, 9 * cm))
for color, df,ax, label in zip(["green", "red"], [firm_flex_df_orig, firm_flex_df], axs, ["c1.25", "c1"]):
    # Plot individual load duration curves
    for col in df.columns:
        sns.lineplot(data=df[col], ax=ax, color=color, alpha=0.4, linewidth=0.5)

    # Plot mean load duration curve
    sns.lineplot(data=df.mean(axis="columns"), ax=ax, color="purple", linewidth=2, label=f"Mean {label}")

    # Plot fill between 10th and 90th percentiles
    ax.fill_between(
        df.index,
        df.quantile(0.1, axis=1),
        df.quantile(0.9, axis=1),
        color=color,
        alpha=0.6,
        label=f"10th-90th percentile {label}"
    )
    ax.set_ylabel("All dispatch [GW]");
    ax.set_xlabel("Hours");
    ax.set_xlim(0,200);
    ax.set_ylim(300,610);
    ax.set_title(label)
    ax.legend();



# %%
# Plot all flexibility
fig, axs = plt.subplots(1, 2, figsize=(30 * cm, 9 * cm))
for color, df,ax, label in zip(["green", "red"], [firm_flex_df_orig, firm_flex_df], axs, ["original", "c1"]):
    # Plot individual load duration curves
    for col in df.columns:
        sns.lineplot(data=df[col], ax=ax, color=color, alpha=0.4, linewidth=0.5)

    # Plot mean load duration curve
    sns.lineplot(data=df.mean(axis="columns"), ax=ax, color="purple", linewidth=2, label=f"Mean {label}")

    # Plot fill between 10th and 90th percentiles
    ax.fill_between(
        df.index,
        df.quantile(0.1, axis=1),
        df.quantile(0.9, axis=1),
        color=color,
        alpha=0.6,
        label=f"10th-90th percentile {label}"
    )
    ax.set_ylabel("All dispatch [GW]");
    ax.set_xlabel("Hours");
    ax.set_xlim(200,1000);
    ax.set_ylim(200,400);
    ax.set_title(label)
    ax.legend();



# %%
# Plot all flexibility
fig, axs = plt.subplots(1, 2, figsize=(30 * cm, 9 * cm))
for color, df,ax, label in zip(["green", "red"], [firm_flex_df_orig, firm_flex_df], axs, ["original", "c1"]):
    # Plot individual load duration curves
    for col in df.columns:
        sns.lineplot(data=df[col], ax=ax, color=color, alpha=0.4, linewidth=0.5)

    # Plot mean load duration curve
    sns.lineplot(data=df.mean(axis="columns"), ax=ax, color="purple", linewidth=2, label=f"Mean {label}")

    # Plot fill between 10th and 90th percentiles
    ax.fill_between(
        df.index,
        df.quantile(0.1, axis=1),
        df.quantile(0.9, axis=1),
        color=color,
        alpha=0.6,
        label=f"10th-90th percentile {label}"
    )
    ax.set_ylabel("All dispatch [GW]");
    ax.set_xlabel("Hours");
    ax.set_xlim(1000,8760);
    ax.set_ylim(0,300);
    ax.set_title(label)
    ax.legend();



# %% [markdown]
# # Others

# %%
# plot_optimal_costs(
#     opt_networks_c1
# )

# %%

# periods_t = sensitivity_periods[(90, 'c1.0', 'Co2L0.0')]
# periods_t.start = periods_t.start.dt.tz_localize(None)
# periods_t.end = periods_t.end.dt.tz_localize(None)
# periods_t.peak_hour = periods_t.peak_hour.dt.tz_localize(None)

# # Periods that are no lonegr identified: 8, 9, 20, 23, 33, 36, 37

# new_costs = pd.DataFrame().astype(float)
# for i, period in periods_t.loc[[8, 9, 20, 23, 33, 36, 37]].iterrows():
#     net_year = get_net_year(period.start)
#     new_costs.loc[i, "costs"] = total_costs[net_year].loc[period.start:period.end].sum()/1e9

# trans_costs = pd.DataFrame().astype(float)
# for i, period in periods.iterrows():
#     net_year = get_net_year(period.start)
#     n = opt_networks_c1[net_year]
#     price_nodes = n.buses[n.buses.carrier == "AC"].index
#     trans_costs.loc[i, "trans_costs"] = (n.buses_t.marginal_price.loc[period.start:period.end, price_nodes] * n.loads_t.p_set.loc[period.start:period.end] ).sum().sum()/1e9
#     trans_costs.loc[i, "costs"] = total_costs[net_year].loc[period.start:period.end].sum()/1e9
# for j, p in periods_t.loc[[8, 9, 20, 23, 33, 36, 37]].iterrows():
#     net_year = get_net_year(p.start)
#     n = opt_networks_c1[net_year]
#     trans_costs.loc["New " + str(j), "trans_costs"] = (n.buses_t.marginal_price.loc[p.start:p.end, price_nodes] * n.loads_t.p_set.loc[p.start:p.end]).sum().sum()/1e9
#     trans_costs.loc["New " + str(j), "costs"] = total_costs[net_year].loc[p.start:p.end].sum()/1e9


# fig, ax = plt.subplots()

# seaborn.scatterplot(data=trans_costs, x="costs", y="trans_costs", ax=ax, color="red")



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
for cost in cost_thresholds:
    sensitivity_periods = sens_periods[cost]

# %%
sens_periods[50][(90, 'c1.25', 'Co2L0.01')]

# %% [markdown]
# # Comparison of gas vs fuel cells
# 
# Compare the sensitivity network with allowed 1% of CO2 emissions. This in the optimal case is used for approx. 2.5% of gas generation in the mix. The SDEs account for 11-40% (24% mean) of annual gas usage and on average utilise approx. 70 GW of gas generation (in peaks up to 195 GW, which is the total capacity in all years, based on p_nom_min). The peak cap. per event varies between 135 and 195 GW suggesting different magnitudes of the events (70-95% of installed cap.).
# In such a system our events recover 9-35% (19% on avg) of gas capital costs, which is usually less than the share in annual gas usage. Also in none of the years with SDEs do we recover all capital costs in gas (max. 95%) and in some years as little as 44%.

# %%
config_gas, scenario_def_gas, years, opt_networks_gas = load_opt_networks("stressful-weather-sensitivities", config_str = "base_s_90_elec_lc1.25_Co2L0.01", load_networks=regenerate_data)

# %%
if regenerate_data:
    caps_gas_fc = pd.DataFrame(index=years, columns=["Gas turbines", "Fuel cells"]).astype(float)
    for year, n in opt_networks_gas.items():
        caps_gas_fc.loc[year, "Gas turbines"] = n.generators[n.generators.carrier.isin(["CCGT", "OCGT"])].p_nom_opt.sum()
        caps_gas_fc.loc[year, "Fuel cells"] = (n.links[n.links.carrier == "H2 Fuel Cell"].p_nom_opt * n.links[n.links.carrier == "H2 Fuel Cell"].efficiency).sum()
    if overwrite_data:
        (caps_gas_fc/1e3).round(0).to_csv("sensitivity_analysis/gas_fc_caps.csv")
else:
    caps_gas_fc = pd.read_csv("sensitivity_analysis/gas_fc_caps.csv", index_col=0)

# %%
opt_obj = pd.read_csv("./processing_data/stressful-weather/opt_objs.csv", index_col=0)
if regenerate_data:
    opt_obj_gas, _ = optimal_costs(opt_networks_gas)
    if overwrite_data:
        opt_obj_gas.to_csv("sensitivity_analysis/opt_objs_gas.csv")
else:
    opt_obj_gas = pd.read_csv("sensitivity_analysis/opt_objs_gas.csv", index_col=0)

# %%
((opt_obj_gas - opt_obj)/1e9).astype(float).describe().round(1)

# %%
if regenerate_data:
    storage_caps = pd.DataFrame(index = years, columns = ["H2", "H2 fuel cell", "H2 electrolysis", "battery", "battery discharger"])

    for year in years:
        for col in storage_caps.columns:
            if col in ["H2", "battery"]:
                storage_caps.loc[year, col] = opt_networks_gas[year].stores[opt_networks_gas[year].stores.carrier == col].sum().e_nom_opt/1e3 # GWh
            else:
                storage_caps.loc[year, col] = opt_networks_gas[year].links[opt_networks_gas[year].links.carrier == col].sum().p_nom_opt/1e3 # GW
    storage_caps = storage_caps.astype(float).round(1)
    if overwrite_data:
        storage_caps.to_csv("sensitivity_analysis/storage_caps_gas.csv")
else:
    storage_caps = pd.read_csv("sensitivity_analysis/storage_caps_gas.csv", index_col=0)

# %%
if regenerate_data:
    cap_renew = pd.DataFrame(index=years, columns=["offwind", "onwind", "solar"]).astype(float)
    for year, n in opt_networks_gas.items():
        cap_renew.loc[year, "offwind"] = n.generators[n.generators.carrier.isin(['offwind-dc','offwind-ac','offwind-float'])].p_nom_opt.sum()
        cap_renew.loc[year, "onwind"] = n.generators[n.generators.carrier == "onwind"].p_nom_opt.sum()
        cap_renew.loc[year, "solar"] = n.generators[n.generators.carrier.isin(['solar','solar-hsat'])].p_nom_opt.sum()
    if overwrite_data:
        cap_renew.to_csv("sensitivity_analysis/renew_caps_gas.csv")
else:
    cap_renew = pd.read_csv("sensitivity_analysis/renew_caps_gas.csv", index_col=0)


# %%
(cap_renew/1e3).describe().round(0)

# %%
# Usage of gas turbines during extreme events.
if regenerate_data:
    usage_of_gas = pd.DataFrame(index = periods.index, columns = ["avg. gas discharge", "max. gas discharge", "avg gas CF", "peak gas CF", "usage of gas", "cost recovery SDE", "annual cost recovery", "annual gas CF"]).astype(float)
    for i, period in periods.iterrows():
        start = period.start
        end = period.end
        peak_hour = period.peak_hour

        n =opt_networks_gas[get_net_year(start)]
        price_nodes = n.buses[n.buses.carrier == "AC"].index
        prices = n.buses_t.marginal_price.loc[:, price_nodes]

        gas_i = n.generators[n.generators.carrier.isin(["CCGT", "OCGT"])].index

        gas_p = n.generators_t.p[gas_i]
        gas_p_helper = gas_p.copy()
        gas_p_helper.columns = gas_i.map(n.generators.bus)
        gas_income = gas_p_helper * prices[gas_p_helper.columns]
        gas_p *= 1e-3
        gas_caps = n.generators.p_nom_opt[gas_i] 
        gas_costs = gas_caps * n.generators.capital_cost[gas_i]
        gas_caps /= 1e3

        usage_of_gas.loc[i, "avg. gas discharge"] = ((gas_p.loc[start:end].mean(axis=0).sum())).round(0)
        usage_of_gas.loc[i, "max. gas discharge"] = (gas_p.loc[start:end].max(axis=0).sum()).round(0)

        usage_of_gas.loc[i, "avg gas CF"] = (usage_of_gas.loc[i, "avg. gas discharge"] / gas_caps.sum()).round(2)
        usage_of_gas.loc[i, "peak gas CF"] = (usage_of_gas.loc[i, "max. gas discharge"] / gas_caps.sum()).round(2)

        usage_of_gas.loc[i, "usage of gas"] = (gas_p.loc[start:end].sum().sum()/gas_p.sum().sum()).round(2)

        usage_of_gas.loc[i, "cost recovery SDE"] = (gas_income.loc[start:end].sum().sum()/gas_costs.sum()).round(2)

        usage_of_gas.loc[i, "annual cost recovery"] = (gas_income.sum().sum()/gas_costs.sum()).round(2)

        usage_of_gas.loc[i, "annual gas CF"] = ((gas_p.sum().sum())/(8760* gas_caps.sum())).round(3)
    if overwrite_data:
        usage_of_gas.to_csv("sensitivity_analysis/usage_of_gas.csv")
else:
    usage_of_gas = pd.read_csv("sensitivity_analysis/usage_of_gas.csv", index_col=0)



# %%
usage_of_gas.describe().round(2)

# %%
if regenerate_data:
    net_load_gas = compute_net_load(opt_networks_gas, opt_networks_gas[1941])
    carrier_tech = ["biomass", "nuclear", "offwind", "solar", "onwind", "ror", "OCGT", "CCGT"]
    links_tech = ["H2 fuel cell", "battery discharger"]
    su_tech = ["PHS", "hydro"]

    peak_gen = pd.DataFrame(columns = carrier_tech + links_tech + su_tech + ["net load"], index = periods.peak_hour).astype(float)

    for period in periods.index:
        peak_hour = periods.loc[period, "peak_hour"]
        net_year = get_net_year(peak_hour)
        n = opt_networks_gas[net_year]
        
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
        peak_gen.loc[peak_hour,"net load"] = net_load_gas.loc[peak_hour]

    peak_gen /= 1e3 # in GW
    peak_gen = peak_gen.round(1)

    if overwrite_data:
        net_load_gas.to_csv("sensitivity_analysis/net_load_gas.csv")
        peak_gen.to_csv("sensitivity_analysis/peak_gen_gas.csv")
else:
    net_load_gas = pd.read_csv("sensitivity_analysis/net_load_gas.csv", index_col=0, parse_dates=True)
    peak_gen = pd.read_csv("sensitivity_analysis/peak_gen_gas.csv", index_col=0, parse_dates=True)

# %%
peak_gen.describe().round(1)

# %%
# Plot a stacked bar plot of the peak generation per technology for each event.
fig, ax = plt.subplots(1,1, figsize = (18*cm, 9*cm))
reindexed_peak_gen = peak_gen.reset_index()[["biomass", "nuclear", "ror", "battery discharger", "hydro", "PHS", "H2 fuel cell", "OCGT","CCGT", "solar", "offwind", "onwind"]]

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
ax.plot(reindexed_peak_gen.index, peak_gen["net load"], color="black", linewidth=0, marker = "_", label="Net load")
ax.legend(bbox_to_anchor=(0, -0.1), loc=2, ncol=4, borderaxespad=0.0, frameon=False, labelspacing=0.75);
ax.set_title("Generation in peak hour per SDE");


# %%
# Plot a stacked bar plot of the peak generation per technology for each event. WITHOUT BATTERY
fig, ax = plt.subplots(1,1, figsize = (18*cm, 9*cm))
reindexed_peak_gen = peak_gen.reset_index()[["biomass", "nuclear", "ror", "hydro", "PHS", "OCGT", "CCGT"]]


reindexed_peak_gen.plot.bar(
    stacked=True,
    ax=ax,
    width = 0.8,
    color=[colours[tech] for tech in reindexed_peak_gen.columns],
);
# Draw a red line for the net load.
ax.legend(bbox_to_anchor=(0, -0.1), loc=2, ncol=4, borderaxespad=0.0, frameon=False, labelspacing=0.75);
ax.set_title("Firm generation in peak hour per SDE");


# %%
# Compute load duration curves of gas turbines.
if regenerate_data:
    gas_i = opt_networks_gas[1941].generators[opt_networks_gas[1941].generators.carrier.isin(["CCGT", "OCGT"])].index
    gen_gas_df = pd.DataFrame(index = range(8760))
    for year, n in opt_networks_gas.items():
        gen_gas_df[year] = n.generators_t.p[gas_i].sum(axis=1).sort_values(ascending=False).values
    gen_gas_df = (gen_gas_df/1e3).round(1)
    if overwrite_data:
        gen_gas_df.to_csv("sensitivity_analysis/gen_gas_df.csv")
else:
    gen_gas_df = pd.read_csv("sensitivity_analysis/gen_gas_df.csv", index_col=0)


# %%
gen_fc_df = pd.read_csv(f"processing_data/{config_name}/gen_fc_df.csv", index_col=0)

# %%
fig, ax = plt.subplots(1, 1, figsize=(18 * cm, 9 * cm))

# Plot individual load duration curves
for col in gen_gas_df.columns:
    sns.lineplot(data=gen_gas_df[col], ax=ax, color="grey", alpha=0.1, linewidth=0.5)


# Plot mean load duration curve
sns.lineplot(data=gen_gas_df.mean(axis="columns"), ax=ax, color="red", linewidth=1, label="Mean")

# Plot fill between 10th and 90th percentiles
ax.fill_between(
    gen_gas_df.index,
    gen_gas_df.quantile(0.1, axis=1),
    gen_gas_df.quantile(0.9, axis=1),
    color="blue",
    alpha=0.3,
    label="10th-90th percentile"
)

ax.set_ylabel("Gas dispatch [GW]")
ax.set_xlabel("Hours")
ax.set_xlim(0, 1000)
ax.set_ylim(0, 200)
# Add for each of the xticklabels and yticklabels the capacity factor (relative to 194 GW installed capacity) as well as the duration relative to 8760 hours.
xticks = ax.get_xticks()
yticks = ax.get_yticks()
xticklabels = [f"{xtick:.0f} \n {int(xtick/8760*100)}%" for xtick in xticks]
yticklabels = [f"{ytick:.0f} ({ytick/194:.0%})" for ytick in yticks]
ax.set_xticklabels(xticklabels)
ax.set_yticklabels(yticklabels)
ax.legend()
plt.show()

# %%
fig, axs = plt.subplots(1, 2, figsize=(30 * cm, 9 * cm))

for ax in axs:
    # Plot individual load duration curves
    for col in gen_gas_df.columns:
        sns.lineplot(data=gen_gas_df[col], ax=ax, color="grey", alpha=0.1, linewidth=0.5)

    # Plot mean load duration curve
    sns.lineplot(data=gen_gas_df.mean(axis="columns"), ax=ax, color="red", linewidth=1, label="Mean")

    # Plot fill between 10th and 90th percentiles
    ax.fill_between(
        gen_gas_df.index,
        gen_gas_df.quantile(0.1, axis=1),
        gen_gas_df.quantile(0.9, axis=1),
        color="blue",
        alpha=0.3,
        label="10th-90th percentile"
    )
    if ax == axs[1]:
        for col in gen_fc_df.columns:
            sns.lineplot(data=gen_fc_df[col], ax=ax, color="grey", alpha=0.1, linewidth=0.5)
        sns.lineplot(data=gen_fc_df.mean(axis="columns"), ax=ax, color="green", linewidth=1, label="FC Mean")
        ax.fill_between(
            gen_fc_df.index,
            gen_fc_df.quantile(0.1, axis=1),
            gen_fc_df.quantile(0.9, axis=1),
            color="green",
            alpha=0.3,
            label="FC 10th-90th percentile"
        )
    ax.set_xlabel("Hours")
    ax.set_xlim(0, 600)
    ax.set_ylim(0, 200)
    # Add for each of the xticklabels and yticklabels the capacity factor (relative to 194 GW installed capacity) as well as the duration relative to 8760 hours.
    if ax == axs[0]:
        ax.set_ylabel("Gas dispatch [GW]")
        yticks = ax.get_yticks() 
        yticklabels = [f"{ytick:.0f} ({ytick/194:.0%})" for ytick in yticks]
        
        ax.set_yticklabels(yticklabels)
    else:
        xticks = ax.get_xticks()
        xticklabels = [f"{xtick:.0f} \n {int(xtick/8760*100)}%" for xtick in xticks]
        ax.set_xticklabels(xticklabels)
        ax.set_ylabel("Gas/fuel cell dispatch [GW]")
    ax.legend()
plt.show()

# %%
# Compute load duration curves of firm flexibility: nuclear, biomass, hydro, PHS
if regenerate_data:
    firm_gen_i = opt_networks_gas[1941].generators[opt_networks_gas[1941].generators.carrier.isin(["nuclear", "biomass", "ror"])].index
    firm_su_i = opt_networks_gas[1941].storage_units[opt_networks_gas[1941].storage_units.carrier.isin(["hydro", "PHS"])].index

    firm_gen_df = pd.DataFrame(index = range(8760))
    for year, n in opt_networks_gas.items():
        df_helper = pd.DataFrame(index = range(8760), columns=["prod"])
        df_helper["prod"] = n.generators_t.p[firm_gen_i].sum(axis=1).values
        df_helper["prod"] += n.storage_units_t.p[firm_su_i].sum(axis=1).values
        firm_gen_df[year] = df_helper.sort_values(by="prod", ascending=False).values
    firm_gen_df = (firm_gen_df/1e3).round(1)
    if overwrite_data:
        firm_gen_df.to_csv("sensitivity_analysis/gas_firm_gen_df.csv")
else:
    firm_gen_df = pd.read_csv("sensitivity_analysis/gas_firm_gen_df.csv", index_col=0)


# %%
firm_gen_df_orig = pd.read_csv(f"processing_data/{config_name}/firm_gen_df.csv", index_col=0)

# %%
# Plot firm flexibility: nuclear, biomass, ror, hydro, PHS
fig, axs = plt.subplots(1, 2, figsize=(30 * cm, 9 * cm),sharey=True)

for ax, df in zip(axs, [firm_gen_df_orig, firm_gen_df]):
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



# %%
# Compute load duration curves of firm flexibility: nuclear, biomass, hydro, PHS
if regenerate_data:
    firm_gen_i = opt_networks_gas[1941].generators[opt_networks_gas[1941].generators.carrier.isin(["nuclear", "biomass", "ror"])].index
    firm_su_i = opt_networks_gas[1941].storage_units[opt_networks_gas[1941].storage_units.carrier.isin(["hydro", "PHS"])].index
    firm_s_i = opt_networks_gas[1941].links[opt_networks_gas[1941].links.carrier.isin(["battery discharger"])].index

    firm_daily_df = pd.DataFrame(index = range(8760))
    for year, n in opt_networks_gas.items():
        df_helper = pd.DataFrame(index = range(8760), columns=["prod"])
        df_helper["prod"] = n.generators_t.p[firm_gen_i].sum(axis=1).values
        df_helper["prod"] += n.storage_units_t.p[firm_su_i].sum(axis=1).values
        df_helper["prod"] -= n.links_t.p1[firm_s_i].sum(axis=1).values
        firm_daily_df[year] = df_helper.sort_values(by="prod", ascending=False).values
    firm_daily_df = (firm_daily_df/1e3).round(1)
    if overwrite_data:
        firm_daily_df.to_csv("sensitivity_analysis/gas_firm_daily_df.csv")
else:
    firm_daily_df = pd.read_csv("sensitivity_analysis/gas_firm_daily_df.csv", index_col=0)


# %%
firm_daily_df_orig = pd.read_csv(f"processing_data/{config_name}/firm_daily_df.csv", index_col=0)

# %%
# Plot daily flexibility: nuclear, biomass, hydro, PHS, BATTERY
fig, ax = plt.subplots(1, 1, figsize=(18 * cm, 9 * cm), sharey=True)

for color, df, label in zip(["green", "red"], [firm_daily_df_orig, firm_daily_df], ["original", "gas"]):
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
    firm_gen_i = opt_networks_gas[1941].generators[opt_networks_gas[1941].generators.carrier.isin(["nuclear", "biomass", "ror", "OCGT", "CCGT"])].index
    firm_su_i = opt_networks_gas[1941].storage_units[opt_networks_gas[1941].storage_units.carrier.isin(["hydro", "PHS"])].index
    firm_s_i = opt_networks_gas[1941].links[opt_networks_gas[1941].links.carrier.isin(["battery discharger"])].index

    firm_flex_df = pd.DataFrame(index = range(8760))
    for year, n in opt_networks_gas.items():
        df_helper = pd.DataFrame(index = range(8760), columns=["prod"])
        df_helper["prod"] = n.generators_t.p[firm_gen_i].sum(axis=1).values
        df_helper["prod"] += n.storage_units_t.p[firm_su_i].sum(axis=1).values
        df_helper["prod"] -= n.links_t.p1[firm_s_i].sum(axis=1).values
        firm_flex_df[year] = df_helper.sort_values(by="prod", ascending=False).values
    firm_flex_df = (firm_flex_df/1e3).round(1)
    if overwrite_data:
        firm_flex_df.to_csv("sensitivity_analysis/gas_firm_flex_df.csv")
else:
    firm_flex_df = pd.read_csv("sensitivity_analysis/gas_firm_flex_df.csv", index_col=0)


# %%
firm_flex_df_orig = pd.read_csv(f"processing_data/{config_name}/firm_flex_df.csv", index_col=0)

# %%
# Plot all flexibility
fig, axs = plt.subplots(1, 2, figsize=(30 * cm, 9 * cm))
for color, df,ax, label in zip(["green", "red"], [firm_flex_df_orig, firm_flex_df], axs, ["original", "gas"]):
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
for color, df,ax, label in zip(["green", "red"], [firm_flex_df_orig, firm_flex_df], axs, ["original", "gas"]):
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
for color, df,ax, label in zip(["green", "red"], [firm_flex_df_orig, firm_flex_df], axs, ["original", "gas"]):
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



# %%



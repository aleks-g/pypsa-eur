# %%
# Initiation

import pypsa 
import datetime as dt 

import yaml

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.path import Path

mpl.rcParams["figure.dpi"] = 150

import pandas as pd
import numpy as np
import seaborn as sns
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

import logging

# Suppress warnings and info messages from 'pypsa.io'
logging.getLogger("pypsa.io").setLevel(logging.ERROR)

cm = 1 / 2.54  # centimeters in inches



# %%
# import os
# import sys

# import geopandas as gpd

# from scipy.stats import pearsonr
# import cartopy
# import cartopy.crs as ccrs
# from weather_plotting import (
#     weather_map,
#     weather_maps,
#     plot_weather_event,
# )

# # Define named tuple structure for interval with start, end, peak, the
# # snapshots and the network.
# from collections import namedtuple

# Interval = namedtuple("Interval", ["start", "end", "peak", "sns", "n"])

# %% [markdown]
# # Initiation

# %%
regenerate_data = False
overwrite_data = False
config_name = "stressful-weather"

# In order to regenerate data, run `generate_data_for_analysis.py`.

# %%
config, scenario_def, years, opt_networks = load_opt_networks(config_name, load_networks=True)


# %%
periods = load_periods(config)

# %%
# Load all data we might need that is pre-generated in `generate_data_for_analysis.py`.
folder = f"./processing_data/{config_name}"

# Load: net load, total load, winter load
net_load = pd.read_csv(f"{folder}/net_load.csv", index_col=0, parse_dates=True)

nodal_load = pd.read_csv(f"{folder}/nodal_load.csv", index_col=0, parse_dates=True)

# Costs: nodal prices, total electricity/storage/fuel cell costs
all_prices = pd.read_csv(f"{folder}/all_prices.csv", index_col=0, parse_dates=True)

wind_distr_df = pd.read_csv(f"processing_data/{config_name}/wind_distr.csv", index_col=[0,1],)


## SDEs
# Stats for storage behaviour
stores_periods = pd.read_csv(f"{folder}/stores_periods.csv", index_col=0)

# Stats for clustering: net load peak hour, highest net load, avg net load, energy deficit, h2 discharge, max fc discharge, avg rel load, wind cf, wind anom, annual cost
stats_periods = pd.read_csv(f"{folder}/stats_periods.csv", index_col=0)


## SYSTEM VALUES
all_system_anomaly = pd.read_csv(f"{folder}/all_system_anomaly.csv", index_col=0, parse_dates=True)
all_used_flexibility = pd.read_csv(f"{folder}/all_used_flexibility.csv", index_col=0, parse_dates=True)
all_flex_anomaly = pd.read_csv(f"{folder}/all_flex_anomaly.csv", index_col=0, parse_dates=True)



# %%
# # Objective values
# opt_objs = pd.read_csv(f"{folder}/opt_objs.csv", index_col=0)
# reindex_opt_objs = opt_objs.copy().sum(axis="columns") 
# reindex_opt_objs.index=years

# Means
# avg_load = pd.read_csv(f"../results/means/load_1941-2020_100bn_12-336h_90_elec_lc1.25_Co2L.csv", index_col=0, parse_dates=True)
# avg_wind = pd.read_csv(f"../results/means/wind_1941-2020_100bn_12-336h_90_elec_lc1.25_Co2L.csv", index_col=0, parse_dates=True)
# avg_solar = pd.read_csv(f"../results/means/solar_1941-2020_100bn_12-336h_90_elec_lc1.25_Co2L.csv", index_col=0, parse_dates=True)

# total_load = pd.read_csv(f"{folder}/total_load.csv", index_col=0, parse_dates=True)
# winter_load = pd.read_csv(f"{folder}/winter_load.csv", index_col=0)

# total_costs_df = pd.read_csv(f"{folder}/total_costs.csv", index_col=[0,1])
# total_storage_costs_df = pd.read_csv(f"{folder}/total_storage_costs.csv", index_col=[0,1])
# total_fc_costs_df = pd.read_csv(f"{folder}/total_fc_costs.csv", index_col=[0,1])

# Storage: storage capacities, storage levels, average storage levels
# s_caps = pd.read_csv(f"{folder}/s_caps.csv", index_col=0)
# su_soc = pd.read_csv(f"{folder}/state_of_charge.csv", index_col=0, parse_dates=True)
# avg_soc = pd.read_csv(f"{folder}/avg_soc.csv", index_col=0, parse_dates=True)

# Transmission:


# Capacity (factors) for wind and solar; wind distribution in the winter
# wind_caps = pd.read_csv(f"{folder}/wind_caps.csv", index_col=0)
# solar_caps = pd.read_csv(f"{folder}/solar_caps.csv", index_col=0)
# wind_cf = xr.open_dataset(f"processing_data/{config_name}/wind_cf.nc").to_dataframe()
# solar_cf = xr.open_dataset(f"processing_data/{config_name}/solar_cf.nc").to_dataframe()


# ## FLEXIBILITY
# # System: detailed
# all_flex_detailed = pd.read_csv(f"{folder}/all_flex_detailed.csv", index_col=0, parse_dates=True)
# avg_flex_detailed = pd.read_csv(f"{folder}/avg_flex_detailed.csv", index_col=0, parse_dates=True)
# periods_flex_detailed = pd.read_csv(f"{folder}/periods_flex_detailed.csv", index_col=0, parse_dates=True)
# periods_anomaly_flex_detailed = pd.read_csv(f"{folder}/periods_anomaly_flex_detailed.csv", index_col=0, parse_dates=True)
# periods_peak_flex_detailed = pd.read_csv(f"{folder}/periods_peak_flex_detailed.csv", index_col=0, parse_dates=True)
# periods_peak_anomaly_flex_detailed = pd.read_csv(f"{folder}/periods_peak_anomaly_flex_detailed.csv", index_col=0, parse_dates=True)


# # Nodal
# nodal_flex_p = pd.read_csv(f"{folder}/nodal_flex_p.csv", index_col=[0,1])
# nodal_seasonality = pd.read_csv(f"{folder}/nodal_seasonality.csv", index_col=0, parse_dates=True)
# nodal_flex_periods = pd.read_csv(f"{folder}/nodal_flex_periods.csv", index_col=0, parse_dates=True)
# nodal_flex_anomaly_periods = pd.read_csv(f"{folder}/nodal_flex_anomaly_periods.csv", index_col=0, parse_dates=True)
# nodal_peak_anomaly_flex = pd.read_csv(f"{folder}/nodal_peak_anomaly_flex.csv", index_col=0, parse_dates=True)


# System: coarse
#all_flex_coarse = pd.read_csv(f"{folder}/all_flex_coarse.csv", index_col=0, parse_dates=True)
#avg_flex_coarse = pd.read_csv(f"{folder}/avg_flex_coarse.csv", index_col=0, parse_dates=True)
#periods_flex_coarse = pd.read_csv(f"{folder}/periods_flex_coarse.csv", index_col=0, parse_dates=True)
#periods_anomaly_flex_coarse = pd.read_csv(f"{folder}/periods_anomaly_flex_coarse.csv", index_col=0, parse_dates=True)
#periods_peak_flex_coarse = pd.read_csv(f"{folder}/periods_peak_flex_coarse.csv", index_col=0, parse_dates=True)
#periods_peak_anomaly_flex_coarse = pd.read_csv(f"{folder}/periods_peak_anomaly_flex_coarse.csv", index_col=0, parse_dates=True)

# # Rewrite costs, storage costs and fuel cell costs in dictionaries for plotting alter.
# total_costs, total_storage_costs, total_fc_costs = {}, {}, {}
# for year in years:
#     df = total_costs_df.loc[year]
#     df.index = pd.to_datetime(df.index)
#     total_costs[year] = df["0"]
#     df = total_storage_costs_df.loc[year]
#     df.index = pd.to_datetime(df.index)
#     total_storage_costs[year] = df["0"]
#     df = total_fc_costs_df.loc[year]
#     df.index = pd.to_datetime(df.index)
#     total_fc_costs[year] = df["0"]

# %% [markdown]
# # System-defining events

# %% [markdown]
# ## Discharge and fuel cells

# %%
# Make a bar plot sorted by values, with different colours for the type of the event.

characteristics = ["discharge", "relative_discharge", "empty", "max_fc_discharge", "affected_fc_p_lim", "ratio_p_lim", "2w_prior_affected_fc_p_lim", "2w_prior_ratio_p_lim", "2w_after_affected_fc_p_lim", "2w_after_ratio_p_lim"]
pretty_names = {
    "discharge": "Total discharge [GWh]",
    "relative_discharge": "Relative discharge [%]",
    "empty": "Empty stores [%]",
    "max_fc_discharge": "Maximal discharge from fuel cells [GW]",
    "affected_fc_p_lim": "Fuel cells at power limit \nat least once [%]",
    "ratio_p_lim": "Ratio of fuel cells at power limit \nthroughout event [%]",
    "2w_prior_affected_fc_p_lim": "Fuel cells at power limit \nat least once during 2 weeks prior to event[%]",
    "2w_prior_ratio_p_lim": "Ratio of fuel cells at power limit \nthroughout 2w prior [%]",
    "2w_after_affected_fc_p_lim": "Fuel cells at power limit \nat least once during 2w after event [%]",
    "2w_after_ratio_p_lim": "Ratio of fuel cells at power limit \nthroughout 2w after [%]"
}

fig, ax = plt.subplots(len(characteristics), 1, figsize=(17.0*cm, 5*len(characteristics)*cm), gridspec_kw={'hspace': 0.5})


for i, c in enumerate(characteristics):
    stores_periods.sort_values(by=c, ascending=False).plot.bar(
        y=c, 
        ax=ax[i], 
        legend=False, 
        color="green"
    )
    ax[i].set_title(pretty_names[c], fontsize=10)
    # Remove x-ticks
    ax[i].set_xticklabels([])
    # Set y-lims to be 0,1 for all characteristics which ahve values in percent;
    if "%" in pretty_names[c]:
        ax[i].set_ylim(0,100)




# %%
# Plot duration of event vs. maximal discharge per hour vs. total discharge (hydrogen) for each event.

fig, ax = plt.subplots(1,1, figsize = (18*cm, 9*cm))

severity = stores_periods[["start", "end", "peak_hour", "discharge", "max_fc_discharge"]].copy()
severity[["start", "end", "peak_hour"]] = severity[["start", "end", "peak_hour"]].apply(pd.to_datetime)
severity["duration"] = (severity["end"] - severity["start"]).dt.total_seconds() / (24 * 3600) #  in days
severity["month"] = severity["peak_hour"].dt.strftime('%b')

# Plot the maximal discharge per hour against the total discharge.
seaborn.scatterplot(
    x="duration",
    y="max_fc_discharge",
    size="discharge",
    sizes=(40, 400),  # Double the size range
    data=severity,
    alpha=0.8,
    hue="month",
    palette="Paired",  # Change the color scheme to 'viridis'
    ax=ax
);

# Aesthetics
ax.set_xlabel("Duration of event [days]");
ax.set_ylabel("Maximal discharge from fuel cells [GW]");
ax.set_ylim(0,None)
ax.tick_params(axis="both", length=0)
ax.grid(axis="both", linestyle="--", alpha=0.5)

# Remove spines
for loc in ["top", "left", "bottom", "right"]:
    ax.spines[loc].set_visible(False)


# Adjust the legend to spread the values with more spacing
handles, labels = ax.get_legend_handles_labels()

# Rename labels.
for i in labels:
    if i == "month":
        labels[labels.index(i)] = "Month"
    elif i == "discharge":
        labels[labels.index(i)] = "Total discharge [TWh]"

# Reorder handles and labels manually.
handles = [handles[0], handles[1], handles[4], handles[3], handles[2], handles[5:]]
labels = [labels[0], labels[1], labels[4], labels[3], labels[2], labels[5:]]



ax.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0, frameon=False, labelspacing=0.75)
plt.show()


# %%
# Make a bar plot where the months are coloured as above, and compare maximal discharge from fuel cells and total discharge.

fig, ax = plt.subplots(1,1, figsize = (18*cm, 9*cm))

severity_sorted = severity.sort_values(by="max_fc_discharge", ascending=False)
order = severity_sorted.index
seaborn.barplot(
    x=severity_sorted.index,
    y="max_fc_discharge",
    data=severity_sorted,
    alpha=0.8,
    hue="month",
    palette="Paired",  # Change the color scheme to 'viridis'
    order=order,
    ax=ax
);

# Aesthetics
ax.set_xlabel("Event number");
ax.set_ylabel("Maximal discharge from fuel cells [GW]");
ax.set_xticklabels(ax.get_xticklabels(),fontsize=8);

# Remove spines
for loc in ["top", "left", "bottom", "right"]:
    ax.spines[loc].set_visible(False)
ax.tick_params(axis="both", length=0);
ax.grid(axis="y", linestyle="--", alpha=0.5);

# Reorder handles and labels manually.
handles, labels = ax.get_legend_handles_labels()
handles = [handles[0], handles[3], handles[2], handles[1]]
labels = [labels[0], labels[3], labels[2], labels[1]]

ax.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0, frameon=False, labelspacing=0.75);



# Do the same for total discharge.
fig, ax = plt.subplots(1,1, figsize = (18*cm, 9*cm))

severity_sorted = severity.sort_values(by="discharge", ascending=False)
order = severity_sorted.index

seaborn.barplot(
    x=severity_sorted.index,
    y="discharge",
    data=severity_sorted,
    alpha=0.8,
    hue="month",
    palette="Paired",  # Change the color scheme to 'viridis'
    order=order,
    ax=ax
);

# Aesthetics
ax.set_xlabel("Event number");
ax.set_ylabel("Total discharge [TWh]");
ax.set_xticklabels(ax.get_xticklabels(),fontsize=8);

# Remove spines
for loc in ["top", "left", "bottom", "right"]:
    ax.spines[loc].set_visible(False);
ax.tick_params(axis="both", length=0);
ax.grid(axis="y", linestyle="--", alpha=0.5);

# Reorder handles and labels manually.
handles, labels = ax.get_legend_handles_labels()
handles = [handles[0], handles[3], handles[2], handles[1]]
labels = [labels[0], labels[3], labels[2], labels[1]]

ax.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0, frameon=False, labelspacing=0.75);






# %% [markdown]
# ## Flexibility and anomaly

# %%
all_flex_anomaly.max()

# %%
clustered_vals = pd.read_csv(f"clustering/{config_name}/clustered_vals_4.csv", index_col=0)

# %%
periods

# %%
plot_cluster_anomalies(all_flex_anomaly, all_system_anomaly, periods, 4, clustered_vals, plot_all_system=False, resampled="12h", save_fig=False, path_str = f"./plots/{config_name}/events_stackplot_simple", cluster_names=["Several power events", "Energy events", "Extreme power events", "Power events"])

# %%
plot_period_anomalies(all_flex_anomaly, all_system_anomaly, periods, plot_all_system=False, resampled="12h", save_fig= True, path_str = f"./plots/{config_name}/events_stackplot_simple")

# %%
plot_period_anomalies(all_flex_anomaly, all_system_anomaly, periods, save_fig = True, path_str = f"./plots/{config_name}/events_stackplot")

# %% [markdown]
# ## Prices and costs per event

# %% [markdown]
# Tried to compare the costs per event.

# %%
all_costs = nodal_load * all_prices

# %% [markdown]
# ## Peak hours

# %%
if regenerate_data:
    carrier_tech = ["biomass", "nuclear", "offwind", "solar", "onwind", "ror"]
    links_tech = ["H2 fuel cell", "battery discharger"]
    su_tech = ["PHS", "hydro"]

    peak_gen = pd.DataFrame(columns = carrier_tech + links_tech + su_tech + ["net load"], index = periods.peak_hour).astype(float)

    for period in periods.index:
        peak_hour = periods.loc[period, "peak_hour"]
        net_year = get_net_year(peak_hour)
        n = opt_networks[net_year]
        
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
        peak_gen.loc[peak_hour,"net load"] = net_load.loc[peak_hour, "Net load"]

    peak_gen /= 1e3 # in GW
    peak_gen = peak_gen.round(1)
    if overwrite_data:
        peak_gen.to_csv(f"{folder}/peak_gen.csv")
else:
    peak_gen = pd.read_csv(f"{folder}/peak_gen.csv", index_col=0, parse_dates=True)

peak_gen


# %%
peak_gen.describe().round(1)

# %%
if regenerate_data:
    cap_renew = pd.DataFrame(index=years, columns=["offwind", "onwind", "solar"]).astype(float)
    for year, n in opt_networks.items():
        cap_renew.loc[year, "offwind"] = n.generators[n.generators.carrier.isin(['offwind-dc','offwind-ac','offwind-float'])].p_nom_opt.sum()
        cap_renew.loc[year, "onwind"] = n.generators[n.generators.carrier == "onwind"].p_nom_opt.sum()
        cap_renew.loc[year, "solar"] = n.generators[n.generators.carrier.isin(['solar','solar-hsat'])].p_nom_opt.sum()
    if overwrite_data:
        cap_renew.to_csv(f"{folder}/cap_renew.csv")
else:
    cap_renew = pd.read_csv(f"{folder}/cap_renew.csv", index_col=0)

# %%
(cap_renew/1e3).round(0)

# %%
(cap_renew/1e3).describe().round(0)

# %%
# Plot a stacked bar plot of the peak generation per technology for each event.
fig, ax = plt.subplots(1,1, figsize = (18*cm, 9*cm))
reindexed_peak_gen = peak_gen.reset_index()[["biomass", "nuclear", "ror", "battery discharger", "hydro", "PHS", "H2 fuel cell", "solar", "offwind", "onwind"]]

colours["H2 fuel cell"] = colours["fuel_cells"]
colours["battery discharger"] = colours["battery"]
colours["PHS"] = colours["phs"]
colours["offwind"] = "#6895dd"
colours["onwind"] = "#235ebc"
colours["solar"] = "#f9d002"

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
# Plot a stacked bar plot of the peak generation per technology for each event.
fig, ax = plt.subplots(1,1, figsize = (18*cm, 9*cm))
reindexed_peak_gen = peak_gen.reset_index()[["biomass", "nuclear", "ror",  "hydro", "PHS", "battery discharger", "H2 fuel cell", "solar", "offwind", "onwind"]]

reindexed_peak_gen[["battery discharger", "H2 fuel cell",]].plot.bar(
    stacked=True,
    ax=ax,
    width = 0.8,
    color=[colours[tech] for tech in reindexed_peak_gen[["battery discharger", "H2 fuel cell",]].columns],
);
# Draw a red line for the net load.
# ax.plot(reindexed_peak_gen.index, peak_gen["net load"], color="black", linewidth=0, marker = "_", label="Net load")
ax.legend(bbox_to_anchor=(0, -0.1), loc=2, ncol=4, borderaxespad=0.0, frameon=False, labelspacing=0.75);
ax.set_title("Generation in peak hour per SDE");


# %%
# Plot the correlation of peak_gen.

fig, ax = plt.subplots(1,1, figsize = (18*cm, 9*cm))
mask = np.triu(peak_gen.corr())
seaborn.heatmap(peak_gen.corr(), annot=True, mask=mask, cmap="coolwarm", ax=ax, fmt=".2f")

# %%
# Scatter plot 
fig, axs = plt.subplots(1,2, figsize = (18*cm, 9*cm), sharey=True)
seaborn.scatterplot(x="H2 fuel cell", y="net load", data=peak_gen, ax=axs[0], color = colours["H2 fuel cell"]);
seaborn.scatterplot(x="battery discharger", y="net load", data=peak_gen, ax=axs[1], color = colours["battery discharger"]);

# Add correlation to plot.
for i, ax in enumerate(axs):
    corr = peak_gen[["net load", ["H2 fuel cell", "battery discharger"][i]]].corr().iloc[0,1]
    ax.text(0.1, 0.9, f"Correlation: {corr:.2f}", transform=ax.transAxes, fontsize=8)


# %%
# Plot a stacked bar plot of the peak generation per technology for each event. WITHOUT BATTERY
fig, ax = plt.subplots(1,1, figsize = (18*cm, 9*cm))
reindexed_peak_gen = peak_gen.reset_index()[[ "nuclear","biomass", # "ror", 
"hydro", "PHS", ]]

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
# Compute load duration curves of fuel cells
if regenerate_data:
    fc_i = opt_networks[1941].links[opt_networks[1941].links.carrier == "H2 fuel cell"].index
    gen_fc_df = pd.DataFrame(index=range(8760)).astype(float)
    for year, n in opt_networks.items():
        gen_fc_df[year] = -n.links_t.p1[fc_i].sum(axis=1).sort_values(ascending=True).values
    gen_fc_df = (gen_fc_df/1e3).round(1) # in GW
    if overwrite_data:
        gen_fc_df.to_csv(f"{folder}/gen_fc_df.csv")
else:
    gen_fc_df = pd.read_csv(f"{folder}/gen_fc_df.csv", index_col=0)

# %%
fig, ax = plt.subplots(1, 1, figsize=(18 * cm, 9 * cm))

# Plot individual load duration curves
for col in gen_fc_df.columns:
    sns.lineplot(data=gen_fc_df[col], ax=ax, color="grey", alpha=0.1, linewidth=0.5)

# Plot mean load duration curve
sns.lineplot(data=gen_fc_df.mean(axis="columns"), ax=ax, color="red", linewidth=1, label="Mean")

# Plot fill between 10th and 90th percentiles
ax.fill_between(
    gen_fc_df.index,
    gen_fc_df.quantile(0.1, axis=1),
    gen_fc_df.quantile(0.9, axis=1),
    color="blue",
    alpha=0.3,
    label="10th-90th percentile"
)

ax.set_ylabel("FC dispatch [GW]")
ax.set_xlabel("Hours")
ax.set_xlim(0, 1000)
ax.set_ylim(0, 80)
# Add for each of the xticklabels and yticklabels the capacity factor (relative to 194 GW installed capacity) as well as the duration relative to 8760 hours.
xticklabels = [f"{i:.0f} \n {int(i/8760*100)}%" for i in ax.get_xticks()]
ax.set_xticklabels(xticklabels)
ax.legend()
plt.show()

# %%
# Compute load duration curves of firm flexibility: nuclear, biomass, hydro, PHS
if regenerate_data:
    firm_gen_i = opt_networks[1941].generators[opt_networks[1941].generators.carrier.isin(["nuclear", "biomass", "ror"])].index
    firm_su_i = opt_networks[1941].storage_units[opt_networks[1941].storage_units.carrier.isin(["hydro", "PHS"])].index

    firm_gen_df = pd.DataFrame(index = range(8760))
    for year, n in opt_networks.items():
        df_helper = pd.DataFrame(index = range(8760), columns=["prod"])
        df_helper["prod"] = n.generators_t.p[firm_gen_i].sum(axis=1).values
        df_helper["prod"] += n.storage_units_t.p[firm_su_i].sum(axis=1).values
        firm_gen_df[year] = df_helper.sort_values(by="prod", ascending=False).values
    firm_gen_df = (firm_gen_df/1e3).round(1)
    if overwrite_data:
        firm_gen_df.to_csv(f"{folder}/firm_gen_df.csv")
else:
    firm_gen_df = pd.read_csv(f"{folder}/firm_gen_df.csv", index_col=0)


# %%
# Plot firm flexibility
fig, ax = plt.subplots(1, 1, figsize=(18 * cm, 9 * cm))

# Plot individual load duration curves
for col in firm_gen_df.columns:
    sns.lineplot(data=firm_gen_df[col], ax=ax, color="grey", alpha=0.1, linewidth=0.5)

# Plot mean load duration curve
sns.lineplot(data=firm_gen_df.mean(axis="columns"), ax=ax, color="red", linewidth=1, label="Mean")

# Plot fill between 10th and 90th percentiles
ax.fill_between(
    firm_gen_df.index,
    firm_gen_df.quantile(0.1, axis=1),
    firm_gen_df.quantile(0.9, axis=1),
    color="blue",
    alpha=0.3,
    label="10th-90th percentile"
)

ax.set_ylabel("Firm generation [GW]");
ax.set_xlabel("Hours");
ax.set_xlim(0,8760);



# %%
# Compute load duration curve of firm + daily flexibility (including batteries)
if regenerate_data:
    firm_gen_i = opt_networks[1941].generators[opt_networks[1941].generators.carrier.isin(["nuclear", "biomass", "ror"])].index
    firm_su_i = opt_networks[1941].storage_units[opt_networks[1941].storage_units.carrier.isin(["hydro", "PHS"])].index
    firm_s_i = opt_networks[1941].links[opt_networks[1941].links.carrier.isin(["battery discharger"])].index

    firm_daily_df = pd.DataFrame(index = range(8760))
    for year, n in opt_networks.items():
        df_helper = pd.DataFrame(index = range(8760), columns=["prod"])
        df_helper["prod"] = n.generators_t.p[firm_gen_i].sum(axis=1).values
        df_helper["prod"] += n.storage_units_t.p[firm_su_i].sum(axis=1).values
        df_helper["prod"] -= n.links_t.p1[firm_s_i].sum(axis=1).values
        firm_daily_df[year] = df_helper.sort_values(by="prod", ascending=False).values
    firm_daily_df = (firm_daily_df/1e3).round(1)
    if overwrite_data:
        firm_daily_df.to_csv(f"{folder}/firm_daily_df.csv")
else:   
    firm_daily_df = pd.read_csv(f"{folder}/firm_daily_df.csv", index_col=0)


# %%
# Plot daily flexibility
fig, ax = plt.subplots(1, 1, figsize=(18 * cm, 9 * cm))

# Plot individual load duration curves
for col in firm_daily_df.columns:
    sns.lineplot(data=firm_daily_df[col], ax=ax, color="grey", alpha=0.1, linewidth=0.5)

# Plot mean load duration curve
sns.lineplot(data=firm_daily_df.mean(axis="columns"), ax=ax, color="red", linewidth=1, label="Mean")

# Plot fill between 10th and 90th percentiles
ax.fill_between(
    firm_daily_df.index,
    firm_daily_df.quantile(0.1, axis=1),
    firm_daily_df.quantile(0.9, axis=1),
    color="blue",
    alpha=0.3,
    label="10th-90th percentile"
)

ax.set_ylabel("Daily flex [GW]");
ax.set_xlabel("Hours");
ax.set_xlim(0,1000);



# %%
# Compute load duration curve of all flexibility.
if regenerate_data:
    firm_gen_i = opt_networks[1941].generators[opt_networks[1941].generators.carrier.isin(["nuclear", "biomass", "ror", "OCGT", "CCGT"])].index
    firm_su_i = opt_networks[1941].storage_units[opt_networks[1941].storage_units.carrier.isin(["hydro", "PHS"])].index
    firm_s_i = opt_networks[1941].links[opt_networks[1941].links.carrier.isin(["battery discharger", "H2 fuel cell"])].index

    firm_flex_df = pd.DataFrame(index = range(8760))
    for year, n in opt_networks.items():
        df_helper = pd.DataFrame(index = range(8760), columns=["prod"])
        df_helper["prod"] = n.generators_t.p[firm_gen_i].sum(axis=1).values
        df_helper["prod"] += n.storage_units_t.p[firm_su_i].sum(axis=1).values
        df_helper["prod"] -= n.links_t.p1[firm_s_i].sum(axis=1).values
        firm_flex_df[year] = df_helper.sort_values(by="prod", ascending=False).values
    firm_flex_df = (firm_flex_df/1e3).round(1)
    if overwrite_data:
        firm_flex_df.to_csv(f"{folder}/firm_flex_df.csv")
else:
    firm_flex_df = pd.read_csv(f"{folder}/firm_flex_df.csv", index_col=0)


# %%
# Plot full flexibility
fig, ax = plt.subplots(1, 1, figsize=(18 * cm, 9 * cm))

# Plot individual load duration curves
for col in firm_flex_df.columns:
    sns.lineplot(data=firm_flex_df[col], ax=ax, color="grey", alpha=0.1, linewidth=0.5)

# Plot mean load duration curve
sns.lineplot(data=firm_flex_df.mean(axis="columns"), ax=ax, color="red", linewidth=1, label="Mean")

# Plot fill between 10th and 90th percentiles
ax.fill_between(
    firm_flex_df.index,
    firm_flex_df.quantile(0.1, axis=1),
    firm_flex_df.quantile(0.9, axis=1),
    color="blue",
    alpha=0.3,
    label="10th-90th percentile"
)

ax.set_ylabel("Full flex [GW]");
ax.set_xlabel("Hours");
ax.set_xlim(0,200);



# %%
# Plot full flexibility
fig, ax = plt.subplots(1, 1, figsize=(18 * cm, 9 * cm))

# Plot individual load duration curves
for col in firm_flex_df.columns:
    sns.lineplot(data=firm_flex_df[col], ax=ax, color="grey", alpha=0.3, linewidth=0.5)

# Plot mean load duration curve
sns.lineplot(data=firm_flex_df.mean(axis="columns"), ax=ax, color="red", linewidth=1, label="Mean")

# Plot fill between 10th and 90th percentiles
ax.fill_between(
    firm_flex_df.index,
    firm_flex_df.quantile(0.1, axis=1),
    firm_flex_df.quantile(0.9, axis=1),
    color="blue",
    alpha=0.4,
    label="10th-90th percentile"
)

ax.set_ylabel("Full flex [GW]");
ax.set_xlabel("Hours");
ax.set_xlim(0,8760);



# %%
# Plot full flexibility
fig, axs = plt.subplots(1, 3, figsize=(18 * cm, 9 * cm), sharey=True, )

ax = axs[0]

# Plot individual load duration curves
for col in firm_flex_df.columns:
    sns.lineplot(data=firm_flex_df[col], ax=ax, color="grey", alpha=0.2, linewidth=0.5)

# Plot mean load duration curve
sns.lineplot(data=firm_flex_df.mean(axis="columns"), ax=ax, color="red", linewidth=1, label="Mean", legend=False)

# Plot fill between 10th and 90th percentiles
ax.fill_between(
    firm_flex_df.index,
    firm_flex_df.quantile(0.1, axis=1),
    firm_flex_df.quantile(0.9, axis=1),
    color="blue",
    alpha=0.3,
    label="10th-90th percentile"
)

ax.set_ylabel("Daily flex [GW]");
ax.set_xlabel("Hours");
ax.set_xlim(0,100);

ax = axs[1]

# Plot individual load duration curves
for col in firm_flex_df.columns:
    sns.lineplot(data=firm_flex_df[col], ax=ax, color="grey", alpha=0.2, linewidth=0.5)

# Plot mean load duration curve
sns.lineplot(data=firm_flex_df.mean(axis="columns"), ax=ax, color="red", linewidth=1, label="Mean", legend=False)

# Plot fill between 10th and 90th percentiles
ax.fill_between(
    firm_flex_df.index,
    firm_flex_df.quantile(0.1, axis=1),
    firm_flex_df.quantile(0.9, axis=1),
    color="blue",
    alpha=0.3,
    label="10th-90th percentile"
)

ax.set_ylabel("Daily flex [GW]");
ax.set_xlabel("Hours");
ax.set_xlim(100,1000);
ax.set_xticks([100, 200, 500, 1000]);

ax.legend(bbox_to_anchor=(0, -0.2), ncol=2, borderaxespad=0.0, frameon=False, labelspacing=0.75);

ax = axs[2]

# Plot individual load duration curves
for col in firm_flex_df.columns:
    sns.lineplot(data=firm_flex_df[col], ax=ax, color="grey", alpha=0.2, linewidth=0.5)

# Plot mean load duration curve
sns.lineplot(data=firm_flex_df.mean(axis="columns"), ax=ax, color="red", linewidth=1, label="Mean", legend=False)

# Plot fill between 10th and 90th percentiles
ax.fill_between(
    firm_flex_df.index,
    firm_flex_df.quantile(0.1, axis=1),
    firm_flex_df.quantile(0.9, axis=1),
    color="blue",
    alpha=0.3,
    label="10th-90th percentile"
)

ax.set_ylabel("Daily flex [GW]");
ax.set_xlabel("Hours");
ax.set_xlim(1000,8760);
ax.set_xticks([1000, 2000, 5000, 8760]);


# %% [markdown]
# ### Prices

# %%
avg_prices = pd.DataFrame(index = periods.index, columns = all_prices.columns).astype(float)
for period in periods.index:
    start, end = periods.loc[period, ["start", "end"]]
    avg_prices.loc[period] = all_prices.loc[start:end].mean()

# %%
avg_prices_normalized = avg_prices.div(avg_prices.max(axis=1), axis=0)

# %%
display(avg_prices_normalized.T.describe().T[["mean", "std", "50%"]].sort_values("std", ascending=False).head(5))
display(avg_prices_normalized.T.describe().T[["mean", "std", "50%"]].sort_values("std", ascending=False).tail(5))

# %%
avg_prices_normalized.T.std()

# %%
# Plot a scatter plot of std in avg_prices_normalized and annotate which event it is.

# The lower the standard deviation, the more spread out the event is, i.e. the prices are more similar regionally.

fig, ax = plt.subplots(1,1, figsize = (18*cm, 9*cm))

seaborn.scatterplot(
    x=avg_prices_normalized.index,
    y=avg_prices_normalized.T.std(),
    data=avg_prices_normalized,
    alpha=0.8,
    ax=ax
);


# %%
peak_gen

# %%
df = peak_gen["H2 fuel cell"]
df.index = periods.index

# %%
fig, ax = plt.subplots(1,1, figsize = (18*cm, 9*cm))

seaborn.scatterplot(
    y=avg_prices_normalized.T.std(),
    x=df,
    data=avg_prices_normalized,
    alpha=0.8,
    ax=ax
);

# %% [markdown]
# ## Similarities

# %%
# Find years for which the capacity factors are the most similar in the winter.
# For this, take the period from October to March for each networks and compute the daily capacity factors and then approximate a distribution for each year.
# Afterwards compute the Wasserstein distance between the distributions for each year and find the years with the smallest distance.

wind_distr = {y: wind_distr_df.loc[y]["0"].dropna() for y in years}
for df in wind_distr.values():
    df.index = pd.to_datetime(df.index)

# Compute the KDE for each year.
kdes = {y: gaussian_kde(wind_distr[y].dropna(), bw_method=0.1) for y in years}

# Compute the Wasserstein distance between each pair of years.
wasserstein = pd.DataFrame(index=years, columns=years)
for y1 in years:
    for y2 in years:
        wasserstein.loc[y1, y2] = wasserstein_distance(wind_distr[y1].dropna(), wind_distr[y2].dropna())

# Find the five years with the smallest (non zero) Wasserstein distance.
wasserstein.replace(0, np.nan).stack().nsmallest(5)

# %%
# Plot the distributions for the years with the smallest Wasserstein distance.
fig, ax = plt.subplots(5,1, figsize=(32.0*cm, 25*cm), sharex=True)

for i, y in enumerate(wasserstein.replace(0, np.nan).stack().nsmallest(10).index[::2]):
    x = np.linspace(0, 1, 1000)
    ax[i].plot(x, kdes[y[0]](x), label=f"{y[0]}")
    ax[i].plot(x, kdes[y[1]](x), label=f"{y[1]}")
    ax[i].set_title(f"Year {y}")

    # If there are system-defining events, plot the days of the SDEs.
    year1, year2 = y
    for j, row in periods.iterrows():
        if get_year_period(row) == year1:
            days = periods.loc[periods.index[j]]
            for k in range((days["end"] - days["start"]).days + 1):
                list_days = [(days["start"] + dt.timedelta(days=k)).date()]
                list_cfs = wind_distr[year1].loc[list_days]
                ax[i].plot(list_cfs.values, [kdes[y[0]](val) for val in list_cfs.values], color="blue", marker = "x", alpha=0.5)
        elif get_year_period(row) == year2:
            days = periods.loc[periods.index[j]]
            for k in range((days["end"] - days["start"]).days + 1):
                list_days = [(days["start"] + dt.timedelta(days=k)).date()]
                list_cfs = wind_distr[year2].loc[list_days]
                ax[i].plot(list_cfs.values, [kdes[y[1]](val) for val in list_cfs.values], color="orange",marker = "x",  alpha=0.5)
        else:
            continue

            


    ax[i].set_ylabel("Density")
    ax[i].legend()
    ax[-1].set_xlabel("Daily capacity factor")

plt.show();
plt.close();

# %%
# For reference, find the five years with the largest Wasserstein distance and plot them.

fig, ax = plt.subplots(5,1, figsize=(32.0*cm, 25*cm), sharex=True)

for i, y in enumerate(wasserstein.replace(0, np.nan).stack().nlargest(10).index[::2]):
    x = np.linspace(0, 1, 1000)
    ax[i].plot(x, kdes[y[0]](x), label=f"{y[0]}")
    ax[i].plot(x, kdes[y[1]](x), label=f"{y[1]}")
    ax[i].set_title(f"Year {y}")

    # If there are system-defining events, plot the days of the SDEs.
    year1, year2 = y
    for j, row in periods.iterrows():
        if get_year_period(row) == year1:
            days = periods.loc[periods.index[j]]
            for k in range((days["end"] - days["start"]).days + 1):
                list_days = [(days["start"] + dt.timedelta(days=k)).date()]
                list_cfs = wind_distr[year1].loc[list_days]
                ax[i].plot(list_cfs.values, [kdes[y[0]](val) for val in list_cfs.values], color="blue", marker = "x", alpha=0.5)
        elif get_year_period(row) == year2:
            days = periods.loc[periods.index[j]]
            for k in range((days["end"] - days["start"]).days + 1):
                list_days = [(days["start"] + dt.timedelta(days=k)).date()]
                list_cfs = wind_distr[year2].loc[list_days]
                ax[i].plot(list_cfs.values, [kdes[y[1]](val) for val in list_cfs.values], color="orange", marker = "x", alpha=0.5)
        else:
            continue

            


    ax[i].set_ylabel("Density")
    ax[i].legend()
    ax[-1].set_xlabel("Daily capacity factor")

plt.show();
plt.close();

# %% [markdown]
# Recovery of the costs for
# - fuel cells
# - fuel cells and H2 storage
# - fuel cells and H2 infrastructure
# - battery dischargers
# - batteries altogether
# during the SDES:
# 
# 1. Fuel cells recover 91% on average of their costs during the SDEs.
# 2. During SDEs we recover on avg. 68% of fuel cell and H2 energy storage (from 50% to 93%).
# 3. During SDEs we recover around 62% of H2 storage costs - electrolysers do not cost much as capacities are relatively small (and also their capacity factors are much higher.)
# 4. Battery dischargers and chargers recover 93% on average of their costs during the SDEs. Note that battery dischargers in PyPSA-Eur have no capital costs, and only battery chargers have capital cost (their capacities are the same save for efficiency). The variations are much larger.
# 5. Batteries recover 22% on average of their costs during the SDEs. It can be from 8.5% up to 33% but this also refers to the storage.

# %%
if regenerate_data:
    recov_disc = pd.DataFrame(index = periods.index, columns = ["fuel cell", "H2 store + d", "H2", "battery c/d", "battery c/d + store"]).astype(float)
    for i, period in periods.iterrows():
        n = opt_networks[get_net_year(period.start)]
        fc_i = n.links.loc[n.links.carrier == "H2 fuel cell"].index
        batt_i = n.links.loc[n.links.carrier == "battery charger"].index

        elec_i = n.links.loc[n.links.carrier == "H2 electrolysis"].index
        h2_i = n.stores.loc[n.stores.carrier == "H2"].index

        fc_costs = n.links.loc[fc_i, "capital_cost"] * n.links.loc[fc_i, "p_nom_opt"]
        batt_costs = n.links.loc[batt_i, "capital_cost"] * n.links.loc[batt_i, "p_nom_opt"]
        elec_costs = n.links.loc[elec_i, "capital_cost"] * n.links.loc[elec_i, "p_nom_opt"]
        h2_costs = n.stores.loc[h2_i, "capital_cost"] * n.stores.loc[h2_i, "e_nom_opt"]
        elec_costs.index = fc_i
        h2_costs.index = fc_i

        batt_costs_e = (n.stores.capital_cost * n.stores.e_nom_opt).filter(like="battery")
        batt_costs_e.index = batt_i

        fc_p = - n.links_t.p1.loc[:, fc_i] * all_prices.loc[n.snapshots].values
        batt_p = - n.links_t.p1.loc[:, batt_i] * all_prices.loc[n.snapshots].values

        rec_fc = (fc_p.loc[period.start:period.end].sum().sum()/fc_costs.sum())
        rec_fc_h2 = (fc_p.loc[period.start:period.end].sum().sum()/(fc_costs.sum() + h2_costs.sum()))
        rec_h2 = (fc_p.loc[period.start:period.end].sum().sum()/(fc_costs.sum() + elec_costs.sum() + h2_costs.sum()))
        batt = (batt_p.loc[period.start:period.end].sum().sum()/batt_costs.sum())
        batt_e = (batt_p.loc[period.start:period.end].sum().sum()/(batt_costs.sum() + batt_costs_e.sum()))

        recov_disc.loc[i] = [rec_fc, rec_fc_h2, rec_h2, batt, batt_e]
    if overwrite_data:
        recov_disc.to_csv(f"{folder}/recov_disc.csv")
else:
    recov_disc = pd.read_csv(f"{folder}/recov_disc.csv", index_col=0)

display(recov_disc.describe())
display(recov_disc.T)



# %%



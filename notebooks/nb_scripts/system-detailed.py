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
# from scipy.stats import pearsonr
# import cartopy
# import cartopy.crs as ccrs
# import geopandas as gpd


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
config, scenario_def, years, opt_networks = load_opt_networks(config_name, load_networks=regenerate_data)


# %%
periods = load_periods(config)

# %%
# Load all data we might need that is pre-generated in `generate_data_for_analysis.py`.
folder = f"./processing_data/{config_name}"

# Objective values
opt_objs = pd.read_csv(f"{folder}/opt_objs.csv", index_col=0)
reindex_opt_objs = opt_objs.copy().sum(axis="columns") 
reindex_opt_objs.index=years


# Load: net load, total load, winter load
net_load = pd.read_csv(f"{folder}/net_load.csv", index_col=0, parse_dates=True)
total_load = pd.read_csv(f"{folder}/total_load.csv", index_col=0, parse_dates=True)
winter_load = pd.read_csv(f"{folder}/winter_load.csv", index_col=0)
# nodal_load = pd.read_csv(f"{folder}/nodal_load.csv", index_col=0, parse_dates=True)

# Costs: nodal prices, total electricity/storage/fuel cell costs
all_prices = pd.read_csv(f"{folder}/all_prices.csv", index_col=0, parse_dates=True)
total_costs_df = pd.read_csv(f"{folder}/total_costs.csv", index_col=[0,1])
total_storage_costs_df = pd.read_csv(f"{folder}/total_storage_costs.csv", index_col=[0,1])
total_fc_costs_df = pd.read_csv(f"{folder}/total_fc_costs.csv", index_col=[0,1])

# Storage: storage capacities, storage levels, average storage levels
s_caps = pd.read_csv(f"{folder}/s_caps.csv", index_col=0)
su_soc = pd.read_csv(f"{folder}/state_of_charge.csv", index_col=0, parse_dates=True)
avg_soc = pd.read_csv(f"{folder}/avg_soc.csv", index_col=0, parse_dates=True)

# Transmission:

# Annual values: winter wind, annual inflow, annual wind cfs
annual_inflow = pd.read_csv(f"{folder}/annual_inflow.csv", index_col=0)
annual_cfs = pd.read_csv(f"{folder}/annual_cfs.csv", index_col=0)
winter_cfs = pd.read_csv(f"{folder}/winter_cfs.csv", index_col=0)

# Annual stats: longest deficit, highest net load, highest weekly load
longest_deficit = pd.read_csv(f"{folder}/longest_net_load_deficit.csv", index_col=0)
highest_deficit = pd.read_csv(f"{folder}/highest_net_load_deficit.csv", index_col=0)
max_weekly_net_load = pd.read_csv(f"{folder}/max_weekly_net_load.csv", index_col=0)

## SDEs
# Stats for storage behaviour
stores_periods = pd.read_csv(f"{folder}/stores_periods.csv", index_col=0)

# Stats for clustering: net load peak hour, highest net load, avg net load, energy deficit, h2 discharge, max fc discharge, avg rel load, wind cf, wind anom, annual cost
stats_periods = pd.read_csv(f"{folder}/stats_periods.csv", index_col=0)


## FLEXIBILITY
# System: detailed
all_flex_detailed = pd.read_csv(f"{folder}/all_flex_detailed.csv", index_col=0, parse_dates=True)
avg_flex_detailed = pd.read_csv(f"{folder}/avg_flex_detailed.csv", index_col=0, parse_dates=True)
periods_flex_detailed = pd.read_csv(f"{folder}/periods_flex_detailed.csv", index_col=0, parse_dates=True)
periods_anomaly_flex_detailed = pd.read_csv(f"{folder}/periods_anomaly_flex_detailed.csv", index_col=0, parse_dates=True)
periods_peak_flex_detailed = pd.read_csv(f"{folder}/periods_peak_flex_detailed.csv", index_col=0, parse_dates=True)
periods_peak_anomaly_flex_detailed = pd.read_csv(f"{folder}/periods_peak_anomaly_flex_detailed.csv", index_col=0, parse_dates=True)




# Nodal
nodal_flex_p = pd.read_csv(f"{folder}/nodal_flex_p.csv", index_col=[0,1])
nodal_seasonality = pd.read_csv(f"{folder}/nodal_seasonality.csv", index_col=0, parse_dates=True)
nodal_flex_periods = pd.read_csv(f"{folder}/nodal_periods_flex_u.csv", index_col=0, parse_dates=True)
nodal_flex_anomaly_periods = pd.read_csv(f"{folder}/nodal_anomaly_flex_u.csv", index_col=0, parse_dates=True)
nodal_peak_anomaly_flex = pd.read_csv(f"{folder}/nodal_peak_anomaly_flex_u.csv", index_col=0, parse_dates=True)


## SYSTEM VALUES
all_system_anomaly = pd.read_csv(f"{folder}/all_system_anomaly.csv", index_col=0, parse_dates=True)
all_used_flexibility = pd.read_csv(f"{folder}/all_used_flexibility.csv", index_col=0, parse_dates=True)
all_flex_anomaly = pd.read_csv(f"{folder}/all_flex_anomaly.csv", index_col=0, parse_dates=True)



# %%

# Means
# avg_load = pd.read_csv(f"../results/means/load_1941-2020_100bn_12-336h_90_elec_lc1.25_Co2L.csv", index_col=0, parse_dates=True)
# avg_wind = pd.read_csv(f"../results/means/wind_1941-2020_100bn_12-336h_90_elec_lc1.25_Co2L.csv", index_col=0, parse_dates=True)
# avg_solar = pd.read_csv(f"../results/means/solar_1941-2020_100bn_12-336h_90_elec_lc1.25_Co2L.csv", index_col=0, parse_dates=True)

# Capacity (factors) for wind and solar; wind distribution in the winter
# wind_caps = pd.read_csv(f"{folder}/wind_caps.csv", index_col=0)
# solar_caps = pd.read_csv(f"{folder}/solar_caps.csv", index_col=0)
# wind_cf = xr.open_dataset(f"processing_data/{config_name}/wind_cf.nc").to_dataframe()
# solar_cf = xr.open_dataset(f"processing_data/{config_name}/solar_cf.nc").to_dataframe()
# wind_distr_df = pd.read_csv(f"processing_data/{config_name}/wind_distr.csv", index_col=[0,1],)

# System: coarse
#all_flex_coarse = pd.read_csv(f"{folder}/all_flex_coarse.csv", index_col=0, parse_dates=True)
#avg_flex_coarse = pd.read_csv(f"{folder}/avg_flex_coarse.csv", index_col=0, parse_dates=True)
#periods_flex_coarse = pd.read_csv(f"{folder}/periods_flex_coarse.csv", index_col=0, parse_dates=True)
#periods_anomaly_flex_coarse = pd.read_csv(f"{folder}/periods_anomaly_flex_coarse.csv", index_col=0, parse_dates=True)
#periods_peak_flex_coarse = pd.read_csv(f"{folder}/periods_peak_flex_coarse.csv", index_col=0, parse_dates=True)
#periods_peak_anomaly_flex_coarse = pd.read_csv(f"{folder}/periods_peak_anomaly_flex_coarse.csv", index_col=0, parse_dates=True)

# %%
# Rewrite costs, storage costs and fuel cell costs in dictionaries for plotting alter.
total_costs, total_storage_costs, total_fc_costs = {}, {}, {}
for year in years:
    df = total_costs_df.loc[year]
    df.index = pd.to_datetime(df.index)
    total_costs[year] = df["0"]
    df = total_storage_costs_df.loc[year]
    df.index = pd.to_datetime(df.index)
    total_storage_costs[year] = df["0"]
    df = total_fc_costs_df.loc[year]
    df.index = pd.to_datetime(df.index)
    total_fc_costs[year] = df["0"]


# %% [markdown]
# # Clustering

# %%
n_clusters = 5
ranked_heatmap = pd.read_csv(f"clustering/{config_name}/kpi.csv", index_col=0)
clustered_vals = pd.read_csv(f"clustering/{config_name}/clustered_vals_{n_clusters}.csv", index_col=0)
combined_clusters = pd.read_csv(f"clustering/{config_name}/combined_clusters_{n_clusters}.csv", index_col=0)
cluster_centroids = pd.read_csv(f"clustering/{config_name}/centroids_{n_clusters}.csv", index_col=0)
ranked_centroids = pd.read_csv(f"clustering/{config_name}/ranked_centroids_{n_clusters}.csv", index_col=0)

# %% [markdown]
# # Attempt at distinguishing events by power or energy event
# ## Power events
# 
# - **Extreme power event** (Cluster 2: 1 event) Most extreme event (outlier?) by net load.
# - **Several severe power events** (Cluster 0: 9 events) Sequence of severe events with chance of recharging (either going to negative net load or close to no deficit). Strong usage of hydrogen, high net load and fuel cell usage, large deficit. Seemingly more expensive years.
# - **Severe power event** (Cluster 4: 8 events) Wind-driven. Severe power event with high usage of fuel cells, with hydrogen covering large share of deficit, strong wind anomaly. Medium difficulty of years.
# 
# ## Mixed events
# 
# - **Mixed event** (Cluster 3: 8 events) Load-driven. High load (and net load), medium duration with higher usage of fuel cells or longer duration with low H2 discharge. Difficult years?
# 
# ## Energy events
# 
# - **Energy event** (Cluster 1: 15 events) Less severe event, higher energy deficit, less hydrogen discharge or peaking, longer duration. Presumably in less severe years, as less fuel cell capacity is installed.

# %%
# count members of clusters
clustered_vals.cluster.value_counts()

# print the event nr for each cluster
for cluster in range(n_clusters):
    print(f"Cluster {cluster}:")
    print(clustered_vals[clustered_vals.cluster == cluster].index)


# %%
cluster_centroids

# %% [markdown]
# # Cost plots

# %%
# HOURLY VALUE OF ELECTRICITY vs. HOURLY VALUE OF FUEL CELLS
plot_duals(
        periods, 
        years, 
        total_fc_costs, 
        total_costs, 
        mpl.colors.LogNorm(vmin=1, vmax=100, clip=True),
        mpl.colors.LogNorm(vmin=0.1, vmax=10000, clip=True), 
        "Blues",
        "Oranges",
        "Shadow price of fuel cells (mn EUR)",
        "Hourly electricity cost (million EUR)",
        [1, 10, 100],
        [1, 10, 100, 1000, 10000],
        save_fig=True,
        path_str=f"./plots/{config_name}/hourly_value_fc_elec_price"
    )

# %%
# HOURLY VALUE OF STORAGE vs. HOURLY VALUE OF FUEL CELLS
plot_duals(
        periods, 
        years, 
        total_fc_costs, 
        total_storage_costs, 
        mpl.colors.LogNorm(vmin=1, vmax=100, clip=True),
        mpl.colors.LogNorm(vmin=0.1, vmax=100, clip=True), 
        "Blues",
        "Greens",
        "Shadow price of fuel cells (mn EUR)",
        "Hourly value of storage (bn EUR)",
        [1, 10, 100],
        [0.1, 1, 10, 100],
        right_scaling=1e-9,
        save_fig=True,
        path_str=f"./plots/{config_name}/hourly_value_fc_storage_val"
    )

# %%
# HOURLY VALUE OF FUEL CELLS vs. SHARE IN ELEC PRICE
share_fc = {y: (total_fc_costs[y] / total_costs[y]).round(2) for y in years}
plot_duals(
        periods, 
        years, 
        total_fc_costs, 
        total_storage_costs, 
        mpl.colors.LogNorm(vmin=1, vmax=100, clip=True),
        mpl.colors.LogNorm(vmin=1, vmax=20, clip=True), 
        "Blues",
        "Greys",
        "Shadow price of fuel cells (mn EUR)",
        "Share of fuel cell in electricity price [%]",
        [1, 10, 100],
        [1, 10, 20],
        right_scaling=100,
        save_fig=True,
        path_str=f"./plots/{config_name}/hourly_value_fc_share_elec"
    )

# %% [markdown]
# # Flexibility

# %%
avg_flex_detailed.resample("w").mean().plot()

# %%
avg_flex_detailed.describe().round(2)

# %%
periods_peak_flex_detailed.describe().round(2)

# %%
periods_peak_anomaly_flex_detailed.describe().round(2)

# %%
periods_flex_detailed.describe().round(2)

# %%
periods_anomaly_flex_detailed.describe().round(2)

# %%
#avg_flex_coarse.resample("w").mean().plot()

# %% [markdown]
# ## Regional flex

# %%


# %% [markdown]
# ## Flexibility capacities

# %%
nodal_flex_p

# %%
system_flex_p = (nodal_flex_p.unstack().sum(axis="rows") / 1e3).round(1)

fig, ax= plt.subplots(1,1, figsize=(30*cm, 9*cm))
# Plot a stacked bar plot for all system flexibility capacities. Set the level 0 of the index to be columns and level 1 to be the index.

colours["H2 fuel cell"] = colours["fuel_cells"]
colours["battery discharger"] = colours["battery"]
colours["PHS"] = colours["phs"]

df_system_flex = system_flex_p.unstack(level=0)
# Reorder df_system_flex to be of the form baseload, and then the order of how we dispatch during extreme events.
df_system_flex = df_system_flex[["nuclear", "biomass", "hydro", "PHS", "battery discharger", "H2 fuel cell"]]

df_system_flex["battery discharger"]
df_system_flex["H2 fuel cell"]

df_system_flex.plot(
    kind="bar", stacked=True, ax=ax,
    color=[colours[t] for t in df_system_flex.columns],
    width=0.9, zorder=0
)

ax.set_ylabel("Flexibility capacity [GW]");

pretty_labels = ["Nuclear", "Biomass", "Hydro", "PHS", "H2 fuel cell", "Battery discharger"]
ax.legend(pretty_labels,loc="upper right", bbox_to_anchor=(0.9, -0.2), frameon=False, labelspacing=0.75, ncols = 7)

plt.show()

# %%
df_system_flex.describe().round(1)

# %%
# NOTE THESE ARE THE DISCHARGE CAPACITIES IN ELECTRICITY OUTPUT
fig, ax= plt.subplots(1,1, figsize=(30*cm, 9*cm))
# Plot a stacked bar plot for all system flexibility capacities. Set the level 0 of the index to be columns and level 1 to be the index.

df_system_flex[["battery discharger", "H2 fuel cell"]].plot(
    kind="bar", stacked=True, ax=ax,
    color=[colours[t] for t in df_system_flex[["battery discharger", "H2 fuel cell"]].columns],
    width=0.9, zorder=0
)


ax.set_ylabel("Flexibility capacity [GW]");

pretty_labels = ["H2 fuel cell", "Battery discharger"]
ax.legend(pretty_labels,loc="upper right", bbox_to_anchor=(0.9, -0.2), frameon=False, labelspacing=0.75, ncols = 7)

plt.show()

# %%
# NOTE THIS IS THE DISCHARGE CAPACITY

fig, ax= plt.subplots(1,1, figsize=(30*cm, 9*cm))
# Plot a stacked bar plot for all system flexibility capacities. Set the level 0 of the index to be columns and level 1 to be the index.
df_system_flex[["H2 fuel cell"]].plot(
    kind="bar", stacked=True, ax=ax,
    color=[colours[t] for t in df_system_flex[["H2 fuel cell"]].columns],
    width=0.9, zorder=0
)

ax.set_ylabel("Flexibility capacity [GW]");

pretty_labels = ["H2 fuel cell"]
ax.legend(pretty_labels,loc="upper right", bbox_to_anchor=(0.9, -0.2), frameon=False, labelspacing=0.75, ncols = 7)

plt.show()

# %%
# fc_i = n.links.loc[n.links.carrier == "H2 fuel cell"].index
# batt_i = n.links.loc[n.links.carrier == "battery discharger"].index

# fc_p = - n.links_t.p1.loc[:, fc_i] * all_prices.iloc[-8760:].values
# batt_p = - n.links_t.p1.loc[:, batt_i] * all_prices.iloc[-8760:].values

# %%
# (fc_p.loc[periods.iloc[-1].start:periods.iloc[-1].end].sum())/(n.links.loc[fc_i, "capital_cost"] * n.links.loc[fc_i, "p_nom_opt"])

# %%
# (batt_p.loc[periods.iloc[-1].start:periods.iloc[-1].end].sum())/ (n.links.loc[batt_i, "capital_cost"] * n.links.loc[batt_i, "p_nom_opt"])

# %%
# Scatter plot of flexibility capacity vs total system costs
reindex_opt_objs = opt_objs.copy().sum(axis="columns") 
reindex_opt_objs.index=years



fig, ax = plt.subplots(1, 2, figsize=(11*cm, 5*cm), gridspec_kw={'width_ratios': [1, 1], 'wspace': 0.5})
plot_scatter(ax[0], reindex_opt_objs/1e9, df_system_flex.sum(axis="columns"), "Total system costs [bn EUR / a]", "Flexibility capacity [GW]", "Opt networks")
plot_scatter(ax[1], reindex_opt_objs/1e9, df_system_flex[["nuclear", "biomass", "hydro", "PHS", "H2 fuel cell"]].sum(axis="columns"), "Total system costs [bn EUR / a]", "Flex cap w/o batt. [GW]", "Opt networks")

# %% [markdown]
# # System insights

# %% [markdown]
# ## Annual values

# %%
# Generate plot where opt_obj and alt_opt_obj are compared side by side.

cs = ["#c0c0c0", '#bf13a0', '#ace37f', "#70af1d", "#92d123", "#235ebc", "#6895dd", "#f9d002"]

fig, ax = plt.subplots(1,1, figsize=(32.0*cm, 7*cm))
(opt_objs / 1e9).plot.bar(stacked=True, ax=ax, color=cs, width=0.7, position=-0.35)



# Labels
ax.set_xlabel("Weather year")
ax.set_ylabel("Annual system costs [bn EUR / a]")
ax.set_title("Optimal system costs across weather years")

# Legend
handles, labels = ax.get_legend_handles_labels()
ax.legend(reversed(handles[:7]), reversed(labels[:7]), bbox_to_anchor=(0, -0.35), loc='upper left', ncol=3, fontsize=9, frameon=False)

# Ticks, grid
ax.yaxis.set_major_locator(MultipleLocator(50))
ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
ax.yaxis.set_minor_locator(MultipleLocator(10))
ax.yaxis.grid(color='lightgray', linestyle='solid', which='major')
ax.yaxis.grid(color='lightgray', linestyle='dotted', which='minor')



# %%
# Hydropower inflow throughout the years.
fig, ax = plt.subplots(figsize=(32.0*cm, 7*cm))

annual_inflow.plot(ax=ax, label = "Annual inflow", color="darkblue")
# Add horizontal line for average inflow in dotted line.
ax.axhline(annual_inflow["Annual inflow"].mean(), color="lightblue", linestyle="--",  label="1941-2020 average")
# Add line for average of every decade.
for i in range(1941, 2020, 10):
    ax.hlines(annual_inflow.loc[i:i+9,"Annual inflow"].mean(), i, i+9, color="black", lw=0.5, linestyle="--", label=f"{i}-{i+9} average")
    # Add annotation for the decade in the middle
    ax.text(i+4, annual_inflow.loc[i:i+9, "Annual inflow"].mean(), f"{i}-{i+9}", verticalalignment="top", horizontalalignment="center", fontsize=8)
ax.set_xlabel("Weather year");
ax.set_title("Annual hydropower inflow");
ax.set_xlim(1940,2021);

# Legend
labels, handles = ax.get_legend_handles_labels()
ax.legend(labels[:2], handles[:2], loc="upper left", frameon=False)

# %% [markdown]
# Hydro reservoirs are barely emptied altogether (and if so, not at the same time). Upon further investigation during the SDEs, there are only 3 resevoirds (GR0 0, AL0 0 and ES0 1) with a capacity above 1 GWh (so not RS0 0, FR0 6, ME0 0, MK0 0) that have more than 10 hours (out of ca. 7000) where the reservoir is empty. All but two other have 0 or 1 hour where they are empty; all substantial reservoirs are never emptied during SDEs. The one exception seems to be 1982-02-26 07:00:00 with 11 empty reservoirs (mostly in southern Europe/Spain) at the end of the event.

# %%
# Wind capacity factors throughout the years

# Compute capacity factor (Europe-wide) for wind power, onshore wind power and offshore wind power.

fig, ax = plt.subplots(1,1, figsize=(32.0*cm, 7*cm))
annual_cfs.plot(ax=ax, color=["blue", "darkblue", "lightblue","orange"], marker="o", linestyle="--", linewidth=1.0)
# Labels
ax.set_xlabel("Weather year")
ax.set_ylabel("Capacity factor")
ax.set_title("Capacity factors for renewables")

# Legend
handles, labels = ax.get_legend_handles_labels()
ax.legend(reversed(handles), reversed(labels), bbox_to_anchor=(0.3, -0.35), loc='upper left', ncol=3, fontsize=9, frameon=False);

# %%
# Capacity factors between October and March only.
fig, ax = plt.subplots(1,1, figsize=(32.0*cm, 7*cm))
winter_cfs.plot(ax=ax, color=["blue", "darkblue", "lightblue"], marker="o", linestyle="--", linewidth=1.0)

# Labels
ax.set_xlabel("Weather year")
ax.set_ylabel("Capacity factor")
ax.set_title("Capacity factors for wind power (October - March)")

# Legend
handles, labels = ax.get_legend_handles_labels()
ax.legend(reversed(handles), reversed(labels), bbox_to_anchor=(0.3, -0.35), loc='upper left', ncol=3, fontsize=9, frameon=False);

# %%
# Winter load

# Plot the avg European load for the different years.
ig, ax = plt.subplots(1,1, figsize=(32.0*cm, 7*cm))
winter_load.plot(ax=ax, color="black", marker="o", linestyle="--", linewidth=1.0)

# Labels
ax.set_xlabel("Weather year")
ax.set_ylabel("Average European load [GWh/h]")
ax.set_title("Average European load during winter")

# Ticks, grid
ax.yaxis.set_major_locator(MultipleLocator(10))
ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
ax.yaxis.set_minor_locator(MultipleLocator(2))
ax.yaxis.grid(color='lightgray', linestyle='solid', which='major')
ax.yaxis.grid(color='lightgray', linestyle='dotted', which='minor')

ax.xaxis.set_major_locator(MultipleLocator(5))
ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.xaxis.grid(color='lightgray', linestyle='solid', which='major')

# %%
# Relative costs, winter wind and winter load

# Plot relative winter load vs. wind capacity factors (on a different y-axis).
base_year = 1965

fig, ax = plt.subplots(1,1, figsize=(32.0*cm, 7*cm))

ax2 = ax.twinx()
ax.plot(winter_load["load"]/winter_load.loc[base_year, "load"], color="black", marker="o", linestyle="--", linewidth=1.0)
ax.plot(winter_cfs["wind"]/winter_cfs.loc[base_year, "wind"], color="blue", marker="o", linestyle="--", linewidth=1.0)
ax2.plot(reindex_opt_objs/reindex_opt_objs.loc[base_year], color="red", marker="o", linestyle="--", linewidth=1.0)

# Labels
ax.set_xlabel("Weather year")
ax.set_ylabel("Relative average European load/wind");
ax2.set_ylabel("Relative costs");
ax.set_title(f"Relative winter load and wind capacity factors\n (Base year: {base_year}/{base_year+1})");

# Ticks, grid
ax.yaxis.set_major_locator(MultipleLocator(0.05))
ax2.yaxis.set_major_locator(MultipleLocator(0.1))
ax.yaxis.grid(color='lightgray', linestyle='solid', which='major')
ax2.yaxis.grid(color='lightgray', linestyle='dotted', which='major')

ax.set_ylim(0.75,1.25);
ax2.set_ylim(0.75,1.25);

# Legend
handles, labels = ax.get_legend_handles_labels()
ax.legend(["Load", "Wind"], bbox_to_anchor=(0, -0.35), loc='upper left', ncol=2, fontsize=9, frameon=False)
handles, labels = ax2.get_legend_handles_labels()
ax2.legend(["Costs"], bbox_to_anchor=(0, -0.45), loc='upper left', ncol=2, fontsize=9, frameon=False)

# Compute correlation between costs and wind cf, and costs and load. 
correlation_wind = reindex_opt_objs.corr(winter_cfs["wind"])
correlation_load = reindex_opt_objs.corr(winter_load["load"])

ax.text(0.5, -0.35, f"Correlation costs/wind: {correlation_wind:.2f}\nCorrelation costs/load: {correlation_load:.2f}", horizontalalignment="center", verticalalignment="top", transform=ax.transAxes, fontsize=9);


# %%
# Relative costs, hydro inflow

# Plot relative winter load vs. wind capacity factors (on a different y-axis).

base_year = 1965

fig, ax = plt.subplots(1,1, figsize=(32.0*cm, 7*cm))

ax2 = ax.twinx()
ax.plot(annual_inflow["Annual inflow"]/annual_inflow.loc[base_year, "Annual inflow"], color="lightblue", marker="o", linestyle="--", linewidth=1.0)
ax2.plot(reindex_opt_objs/reindex_opt_objs.loc[base_year], color="red", marker="o", linestyle="--", linewidth=1.0)

# Labels
ax.set_xlabel("Weather year")
ax.set_ylabel("Relative annual inflow");
ax2.set_ylabel("Relative costs");
ax.set_title(f"Relative hydro inflow and costs\n (Base year: {base_year}/{base_year+1})");

# Ticks, grid
ax.yaxis.set_major_locator(MultipleLocator(0.05))
ax2.yaxis.set_major_locator(MultipleLocator(0.1))
ax.yaxis.grid(color='lightgray', linestyle='solid', which='major')
ax2.yaxis.grid(color='lightgray', linestyle='dotted', which='major')

ax.set_ylim(0.5,1.25);
ax2.set_ylim(0.75,1.25);

# Legend
handles, labels = ax.get_legend_handles_labels()
ax.legend(["Inflow"], bbox_to_anchor=(0, -0.35), loc='upper left', ncol=2, fontsize=9, frameon=False)
handles, labels = ax2.get_legend_handles_labels()
ax2.legend(["Costs"], bbox_to_anchor=(0, -0.45), loc='upper left', ncol=2, fontsize=9, frameon=False)

# Compute correlation of costs with inflow.
correlation = reindex_opt_objs.corr(annual_inflow["Annual inflow"])
ax.text(0.5, -0.35, f"Correlation costs/inflow: {correlation:.2f}", horizontalalignment="center", verticalalignment="top", transform=ax.transAxes, fontsize=9);






# %% [markdown]
# ## Installations

# %%
# Correlations between system costs, winter wind, winter load

# Scatter plot between total system costs, winter wind capacity factor and also winter load.

fig, ax = plt.subplots(1, 2, figsize=(11*cm, 5*cm), gridspec_kw={'width_ratios': [1, 1], 'wspace': 0.5})
plot_scatter(ax[0], reindex_opt_objs/1e9, winter_load["load"], "Total system costs [bn EUR / a]", "Avg. European load [GWh/h]", "Opt networks")
plot_scatter(ax[1], reindex_opt_objs/1e9, winter_cfs["wind"], "Total system costs [bn EUR / a]", "Avg. winter wind cf", "Opt networks")


# %%
# Correlations between system costs, winter load, winter wind, and hydrogen storage

# Plot capacities of hydrogen storage and fuel cells for the different years.
# Compare this to winter load, winter wind capacity factors and total costs.

fig, ax = plt.subplots(3, 3, figsize=(22*cm, 15*cm), gridspec_kw={'width_ratios': [1, 1, 1], 'wspace': 0.5, 'hspace': 0.5})

plot_scatter(ax[0, 0], reindex_opt_objs/1e9, s_caps["H2_e"], "Total system costs [bn EUR / a]", "H2 storage [GWh]", "Stores")
plot_scatter(ax[0, 1], winter_load["load"], s_caps["H2_e"], "Avg. European load [GWh/h]", "H2 storage [GWh]", "Stores")
plot_scatter(ax[0, 2], winter_cfs["wind"], s_caps["H2_e"], "Capacity factor", "H2 storage [GWh]", "Stores")

plot_scatter(ax[1, 0], reindex_opt_objs/1e9, s_caps["FC_p"], "Total system costs [bn EUR / a]", "Fuel cell [GW]", "Stores")
plot_scatter(ax[1, 1], winter_load["load"], s_caps["FC_p"], "Avg. European load [GWh/h]", "Fuel cell [GW]", "Stores")
plot_scatter(ax[1, 2], winter_cfs["wind"], s_caps["FC_p"], "Capacity factor", "Fuel cell [GW]", "Stores")

plot_scatter(ax[2, 0], reindex_opt_objs/1e9, s_caps["EL_p"], "Total system costs [bn EUR / a]", "Electrolysis [GW]", "Stores")
plot_scatter(ax[2, 1], winter_load["load"], s_caps["EL_p"], "Avg. European load [GWh/h]", "Electrolysis [GW]", "Stores")
plot_scatter(ax[2, 2], winter_cfs["wind"], s_caps["EL_p"], "Capacity factor", "Electrolysis [GW]", "Stores")


# %% [markdown]
# ## System extremes

# %%
# Plot day of the year with the highest residual load.

fig, ax = plt.subplots(1, 1, figsize=(16.0 * cm, 7 * cm))

temp_df = highest_deficit.copy()
# Set temp_df.time to be timestamps.
temp_df["time"] = pd.to_datetime(temp_df["time"])
# Replace the first four strings by the same year, 1941, in "time".
temp_df["time"] = temp_df["time"].apply(lambda x: x.replace(year=1942, month=x.month, day=x.day) if x.month < 7 else x.replace(year=1941, month=x.month, day=x.day))


ax.scatter(temp_df["time"], temp_df["value"]/1e3, color="red", marker="x", label="Highest residual load")
ax.set_xlabel("Time of year")
ax.set_ylabel("Residual load [GW]");


# %%
# Plot the highest residual load for each year.
fig, ax =  plt.subplots(1, 1, figsize=(16.0 * cm, 7 * cm))
(highest_deficit["value"]/1e3).plot(ax=ax, color="red", label="Highest residual load")
ax.set_xlabel("Weather year");
ax.set_ylabel("Max. net load [GW]");

# %%
# Longest deficit per year
fig, ax = plt.subplots(1, 1, figsize=(16.0 * cm, 7 * cm))
(longest_deficit["hours"]/24).plot(ax=ax)
ax.set_xlabel("Year");
ax.set_ylabel("Days");

# %%
# Plot the largest weekly net load (deficit) for each year.
fig, ax = plt.subplots()
(max_weekly_net_load["total"]/1e6).plot(ax=ax)
ax.set_ylabel("Weekly net load [TWh]");

# %%
# Correlate fuel cell, electrolyser, and storage capacities with
# - the longest deficit
# - the highest deficit
# - the largest weekly deficit.

# Scatter plots

fig, ax = plt.subplots(3, 3, figsize=(22*cm, 15*cm), gridspec_kw={'width_ratios': [1, 1, 1], 'wspace': 0.5, 'hspace': 0.75})

plot_scatter(ax[0, 0], s_caps["H2_e"], longest_deficit["hours"]/24, "H2 storage [GWh]", "Days", "Longest net load duration")
plot_scatter(ax[0, 1], s_caps["FC_p"], longest_deficit["hours"]/24, "Fuel cell [GW]", "Days", "Longest net load duration")
plot_scatter(ax[0, 2], s_caps["EL_p"], longest_deficit["hours"]/24, "Electrolysis [GW]", "Days", "Longest net load duration")

plot_scatter(ax[1, 0], s_caps["H2_e"], highest_deficit["value"]/1e3, "H2 storage [GWh]", "Highest deficit [GW]", "Highest net load")
plot_scatter(ax[1, 1], s_caps["FC_p"], highest_deficit["value"]/1e3, "Fuel cell [GW]", "Highest deficit [GW]", "Highest net load")
plot_scatter(ax[1, 2], s_caps["EL_p"], highest_deficit["value"]/1e3, "Electrolysis [GW]", "Highest deficit [GW]", "Highest net load")

plot_scatter(ax[2, 0], s_caps["H2_e"], max_weekly_net_load["total"]/1e6, "H2 storage [GWh]", "Energy deficit [TWh]", "Largest weekly deficit")
plot_scatter(ax[2, 1], s_caps["FC_p"], max_weekly_net_load["total"]/1e6, "Fuel cell [GW]", "Energy deficit [TWh]", "Largest weekly deficit")
plot_scatter(ax[2, 2], s_caps["EL_p"], max_weekly_net_load["total"]/1e6, "Electrolysis [GW]", "Weekly deficit [TWh]", "Largest weekly deficit")

# %% [markdown]
# # Archive

# %% [markdown]
# ## Additional hydro analysis

# %%
# # HOURLY VALUE OF ELECTRICITY vs. HOURLY VALUE OF HYDRO STORAGE
# hydro_costs = pd.read_csv(f"{folder}/hydro_costs.csv", index_col=0, parse_dates=True)
# phs_costs = pd.read_csv(f"{folder}/phs_costs.csv", index_col=0, parse_dates=True)
# hyd_costs = pd.concat([hydro_costs, phs_costs], axis="columns")
# total_hyd_costs = {y: hyd_costs.loc[f"{y}-07-01 00:00":f"{y+1}-06-30 23:00"].sum(axis="columns") for y in years}
# plot_duals(
#         periods, 
#         years, 
#         total_hyd_costs, 
#         total_costs, 
#         mpl.colors.LogNorm(vmin=0.1, vmax=10000, clip=True),
#         mpl.colors.LogNorm(vmin=0.1, vmax=10000, clip=True), 
#         "Blues",
#         "Oranges",
#         "Value of hydro storage (mn EUR)"
#         "Hourly electricity cost (million EUR)",
#         [1, 10, 100, 1000, 10000],
#         [1, 10, 100, 1000, 10000],
#         save_fig=True,
#         path_str=f"../plots/{config_name}/hourly_value_hydro_elec_price"
#     )

# %%
# # Compute correlation between hydro storage costs and electricity costs.
# correlation_phs = {}
# correlation_hydro = {}
# total_hydro_costs = {
#         y: hydro_costs.loc[f"{y}-07-01":f"{y+1}-06-30"].sum(axis="columns") for y in years
# }
# total_phs_costs = {
#         y: phs_costs.loc[f"{y}-07-01":f"{y+1}-06-30"].sum(axis="columns") for y in years
# }

# for y in years:
#     correlation_hydro[y] = total_hydro_costs[y].corr(total_costs[y])
#     correlation_phs[y] = total_phs_costs[y].corr(total_costs[y])
# correlation_hydro = pd.Series(correlation_hydro)
# corr_phs = pd.Series(correlation_phs)

# fig, ax = plt.subplots(1, 1, figsize=(18 * cm, 9 * cm))
# correlation_hydro.plot(ax=ax, label="Hydro storage value")
# corr_phs.plot(ax=ax, label="PHS value")
# ax.set_ylabel("Correlation with electricity costs");
# ax.set_xlabel("Year");
# ax.legend();


# %%
# colours = {
#     "DC":  "#8a1caf",
#     "AC": "#70af1d",
#     "biomass": "#baa741",
#     "nuclear": "#ff8c00",
#     "ror": "#3dbfb0",
#     "fuel_cells": "#c251ae",
#     "battery": "#ace37f",
#     "phs": "#51dbcc",
#     "hydro": "#298c81",
# }



# def plot_hydro_value(
#     periods: pd.DataFrame,
#     hydro_costs: pd.DataFrame,
#     phs_costs: pd.DataFrame,
#     rolling: int = 24,
#     mark_sde: bool = True,
#     window_length: pd.Timedelta = pd.Timedelta("30d"),
#     title: str = "Value of storage",
#     total_costs: pd.DataFrame = None,
#     add_costs: bool = False,
# ):
#     nrows = len(periods) // 4 if len(periods) % 4 == 0 else len(periods) // 4 + 1
#     fig, axs = plt.subplots(nrows=nrows, ncols = 4, figsize=(30 * cm, 50 * cm), gridspec_kw={'hspace': 0.5, 'wspace': 0.3},sharey=True)
#     fig.suptitle(title, fontsize=12)
#     # No vertical space between title and the rest of the plot.
#     fig.subplots_adjust(top=0.95)

#     for i, row in periods.iterrows():
#         window_start = row["start"] - window_length
#         window_end = row["end"] + window_length

#         ax = axs.flatten()[i]
#         ax.plot(
#             hydro_costs.loc[window_start:window_end].rolling(rolling).mean().index,
#             hydro_costs.loc[window_start:window_end].rolling(rolling).mean(),
#             color = "#298c81",
#             label = "Hydro",
#             lw=0.75,
#         )
#         ax.plot(
#             phs_costs.loc[window_start:window_end].rolling(rolling).mean().index,
#             phs_costs.loc[window_start:window_end].rolling(rolling).mean(),
#             color = "#51dbcc",
#             label = "PHS",
#             lw=0.75,
#         )

#         if add_costs:
#             ax2 = ax.twinx()
#             ax2.plot(
#                 total_costs.loc[window_start:window_end].rolling(rolling).mean().index,
#                 total_costs.loc[window_start:window_end].rolling(rolling).mean(),
#                 color = "red",
#                 label = "Costs",
#                 lw=0.5,
#                 alpha=0.7,
#                 ls = "--"
#             )


#         if mark_sde:
#             ax.fill_between(
#                 hydro_costs.loc[row["start"]:row["end"]].rolling(rolling).mean().index,
#                 phs_costs.loc[window_start:window_end].rolling(rolling).mean().min(),
#                 phs_costs.loc[window_start:window_end,].rolling(rolling).mean().max(),
#                 color="gray",
#                 alpha=0.2,
#             )

#         ax.set_title(f"Event {i}")
#         ax.set_ylim(0, None)
#         ax.set_xlim(window_start, window_end)
#         ax.legend(loc="upper right", frameon=False, fontsize=7);
#         if add_costs:
#             ax2.set_ylim(0, None)
#             ax2.set_ylabel("Electricity Costs", fontsize=8)
#             labels, handles = ax.get_legend_handles_labels()
#             labels2, handles2 = ax2.get_legend_handles_labels()
#             ax.legend(labels + labels2, handles + handles2, loc="upper right", frameon=False, fontsize=8)
#             ax2.set_yticklabels([])
        
#         # Only mark beginning of months in x-tickmarkers.
#         ax.xaxis.set_major_locator(mdates.MonthLocator())
#         ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%Y"))


#         #ax.set_xlabel("Time")
#         ax.set_ylabel("Value of storage", fontsize=8)
#     plt.show();
#     plt.close();


# %%
# # Plot hydro and PHS storage value for all events, including a month around the event.

# helper_df = pd.concat(total_costs).droplevel(0)
# helper_df.index = pd.to_datetime(helper_df.index)

# plot_hydro_value(periods, hydro_costs.sum(axis="columns"), phs_costs.sum(axis="columns"), rolling=24, mark_sde=True, window_length=pd.Timedelta("30d"), title="Value of hydro storage", total_costs=helper_df, add_costs=True)




# %% [markdown]
# ## Usage of different technologies during events

# %%
# plot_flex_events(periods, all_flex_detailed, avg_flex_detailed, rolling = 48, tech = ["fuel_cells"], title = "Flexibility usage of fuel cells", save_fig=True, path_str=f"./plots/{config_name}/flex_events_fc")


# %%
# plot_flex_events(periods, all_flex_detailed, avg_flex_detailed, rolling = 48, tech = ["biomass"], title = "Flexibility usage of biomass", save_fig=True, path_str=f"./plots/{config_name}/flex_events_biomass")

# %%
# plot_flex_events(periods, all_flex_detailed, avg_flex_detailed, rolling = 48, tech = ["phs"], title = "Flexibility usage of PHS", save_fig=True, path_str=f"./plots/{config_name}/flex_events_phs")

# %%
# plot_flex_events(periods, all_flex_detailed, avg_flex_detailed, rolling = 48, tech = ["hydro"], title = "Flexibility usage of hydro", save_fig=True, path_str=f"./plots/{config_name}/flex_events_hydro")


# %%
# plot_flex_events(periods, all_flex_detailed, avg_flex_detailed, rolling = 48, tech = ["nuclear"], title = "Flexibility usage of nuclear", save_fig=True, path_str=f"./plots/{config_name}/flex_events_nuclear")


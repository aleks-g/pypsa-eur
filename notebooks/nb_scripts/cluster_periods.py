# %%

# Initiation

import pypsa 
import datetime as dt 
# import os
# import sys
import yaml

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.path import Path

mpl.rcParams["figure.dpi"] = 150

import pandas as pd
import numpy as np

import seaborn as sns
import xarray as xr
from sklearn.cluster import KMeans

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
config_name = "stressful-weather"
config, scenario_def, years, opt_networks = load_opt_networks(config_name, load_networks=False)
periods = load_periods(config)

# %%
stats_periods = pd.read_csv(f"processing_data/{config_name}/stats_periods.csv", index_col=0, parse_dates=True)

# %%
# Compute duration of each period
for i, period in periods.iterrows():
    stats_periods.loc[i, "duration"] = (pd.Timestamp(period.end) - pd.Timestamp(period.start)).total_seconds() / 3600

# %%
folder = f"clustering/{config_name}"
save_fig = True

# %%
stats_periods.columns

# %%
# Plot heat map for stats_periods with seaborn

# Drop net load peak hour (too similar to highest net load) and wind cf (too similar to wind anomaly)

heatmap = stats_periods.drop(columns=["start", "end", "peak_hour", "net_load_peak_hour", "wind_cf", "energy_deficit"])

heatmap = heatmap[["highest_net_load", "avg_net_load", "duration", "h2_discharge", "max_fc_discharge", "avg_rel_load", "wind_anom", "annual_cost"]]

cmaps = [
    "Reds",
    "Reds",
    "Greens",
    "Purples",
    "Purples",
    "coolwarm",
    "coolwarm",
    "Greys"
]

norms = [
    mcolors.Normalize(vmin=361, vmax=537),
    mcolors.Normalize(vmin=175, vmax=358),
    mcolors.Normalize(vmin=27, vmax=314),
    #mcolors.Normalize(vmin=10, vmax=64),
    mcolors.Normalize(vmin=2.5, vmax=17),
    mcolors.Normalize(vmin=17, vmax=76),
    mcolors.Normalize(vmin=0.84, vmax=1.16),
    mcolors.Normalize(vmin=-0.28, vmax=0.3),
    mcolors.Normalize(vmin=133, vmax=170),   
]

fig, axs = plt.subplots(1, 8, figsize=(30 * cm, 30 * cm))

for i, (col, cmap, norm) in enumerate(zip(heatmap.columns, cmaps, norms)):
    sns.heatmap(
        heatmap[[col]],
        ax=axs[i],
        cmap=cmap,
        norm=norm,
        cbar=False,
        annot=True,
        fmt=".2f",
        yticklabels= False if i != 0 else True,
    )
    axs[i].tick_params(axis='x', rotation=90)

    # Set y ticks with the index for the first plot.
    if i == 0:
        axs[i].set_yticklabels(heatmap.index)
        axs[i].tick_params(axis='y', rotation=0)
    else:
        axs[i].set_yticklabels([])

plt.subplots_adjust(wspace=0, hspace=0)

if save_fig:
    plt.savefig(f"{folder}/heatmap_stats_periods.png", bbox_inches="tight")

# %%
# Rank all events according to category
ranked_heatmap = heatmap.rank(ascending=False)
# Replace wind_anom with ascending=True values
ranked_heatmap["wind_anom"] = heatmap["wind_anom"].rank(ascending=True)

# %%

# Define KPIs and corresponding titles for plots
kpis = [
    'highest_net_load', 'avg_net_load', 'duration', 'h2_discharge',
    'max_fc_discharge', 'avg_rel_load',  'wind_anom', 'annual_cost'
]
short_kpis = [
    "HNL", "ANL", "D", "H2", "FC", "Load",  "W anom.", "Cost"]

titles = [
    'Net Load Peak Hour', 'Average Net Load', 'Duration', 'H2 Discharge',
    'Max FC Discharge', 'Average Relative Load', 'Wind Capacity Factor', 'Wind Anomaly', 'Annual cost'
]

# Initialize a figure with a grid
fig, axes = plt.subplots(2, 4, figsize=(24 * cm, 24 * cm))
axes = axes.flatten()  # Flatten for easy iteration

# Color map for distinguishing top performers
color_map = plt.cm.get_cmap('tab10', len(kpis))

for i, (kpi, title) in enumerate(zip(kpis, titles)):
    ax = axes[i]

    for event_index, event in ranked_heatmap.iterrows():
        # Get rank for each event across all KPIs
        ranks = [ranked_heatmap.loc[event_index, k] for k in kpis]
        
        # Highlight top 5 events in color for this KPI, others in grey
        if event_index in ranked_heatmap[kpi].sort_values().head(5).index:
            ax.plot(kpis, ranks, color=color_map(i), linewidth=2, linestyle='-')
        else:
            ax.plot(kpis, ranks, color='grey', linewidth=0.5, linestyle='-',alpha=0.5)
    
    # Customize each subplot
    ax.set_title(title)
    ax.set_ylabel('Rank')
    ax.invert_yaxis()  # Rank 1 is the highest severity
    ax.tick_params(axis='x', rotation=45)  # Optional: Rotate x-axis labels for readability
    ax.set_xticks(range(len(kpis)))
    ax.set_xticklabels(short_kpis)  # Optional: Use short KPI names for x-axis labels

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show();

if save_fig:
    plt.savefig(f"{folder}/kpi.pdf", bbox_inches="tight")


# %%
# Get correlations between KPIs
correlations = heatmap.corr()

fig, ax = plt.subplots(1, 1, figsize=(14 * cm, 8 * cm))
mask = np.triu(np.ones_like(correlations, dtype=bool))
sns.heatmap(correlations, cmap='coolwarm', annot=True, fmt=".2f", cbar=False, mask=mask, ax=ax)
plt.title('Correlation between KPIs');

# %%
# Correlation between ranks of KPIs
fig, ax = plt.subplots(1, 1, figsize=(14 * cm, 8 * cm))
mask = np.triu(np.ones_like(ranked_heatmap.corr(), dtype=bool))
sns.heatmap(ranked_heatmap.corr(), cmap='coolwarm', annot=True, fmt=".2f", cbar=False, mask=mask, ax=ax)
ax.set_title("Correlation between KPI ranks");

# %%
# Correlation between total system costs with all columns
# Clear identification of power vs energy event?
costs_kpi = heatmap.copy()

# Plot a heatmap of the correlations between the KPIs and the total system costs
fig, ax = plt.subplots(1, 1, figsize=(5 * cm, 5 * cm))
sns.heatmap(
    costs_kpi.corr()[["annual_cost"]],
    ax=ax,
    cmap="coolwarm",
    cbar=False,
    annot=True,
    fmt=".2f",
    yticklabels=True,
)

# %%
ranked_costs_kpi = costs_kpi.rank(ascending=False)

# Replace wind_cf and wind_anom with ascending=True values
ranked_costs_kpi["wind_anom"] = costs_kpi["wind_anom"].rank(ascending=True)

ranked_costs_kpi.corr()

# Plot a heatmap of the correlations between the KPIs and the total system costs
fig, ax = plt.subplots(1, 1, figsize=(5 * cm, 5 * cm))
sns.heatmap(
    ranked_costs_kpi.corr()[["annual_cost"]],
    ax=ax,
    cmap="coolwarm",
    cbar=False,
    annot=True,
    fmt=".2f",
    yticklabels=True,
)



# %% [markdown]
# # Clustering by values
# Do not cluster by ranks.

# %%
clustered_vals = heatmap.copy().drop(columns=["annual_cost"])

# Normalize the values
normalized_vals = (clustered_vals - clustered_vals.mean()) / clustered_vals.std()

# %%
# Use elbow method to determine optimal number of clusters
inertia = []
for n in range(2, 11):
    kmeans = KMeans(n_clusters=n, random_state=0).fit(normalized_vals)
    inertia.append(kmeans.inertia_)

fig, ax = plt.subplots(1, 1, figsize=(10 * cm, 10 * cm))
ax.plot(range(2, 11), inertia, marker='o')
ax.set_xlabel('Number of clusters');
ax.set_ylabel('Inertia');



# %%
# Select 5 clusters based on elbow method.
# Define the number of clusters
n_clusters = 4

# Fit the KMeans model
kmeans_vals = KMeans(n_clusters=n_clusters, random_state=0).fit(normalized_vals)

# Add the cluster labels to the DataFrame
clustered_vals['cluster'] = kmeans_vals.labels_

# %%

# Have own colormaps as before
cmaps = [
    "Reds",
    "Reds",
    "Greens",
    "Purples",
    "Purples",
    "coolwarm",
    "coolwarm",
    plt.cm.tab10,
]

norms = [
    mcolors.Normalize(vmin=344, vmax=537),
    mcolors.Normalize(vmin=175, vmax=358),
    mcolors.Normalize(vmin=27, vmax=314),
    mcolors.Normalize(vmin=2.5, vmax=17),
    mcolors.Normalize(vmin=17, vmax=76),
    mcolors.Normalize(vmin=0.84, vmax=1.16),
    mcolors.Normalize(vmin=-0.3, vmax=0.3),
    None,
]


fig, axs = plt.subplots(1, 8, figsize=(30 * cm, 30 * cm))

for i, (col, cmap, norm) in enumerate(zip(clustered_vals.columns, cmaps, norms)):
    sns.heatmap(
        clustered_vals.sort_values("cluster")[[col]],
        ax=axs[i],
        cmap=cmap,
        norm=norm,
        cbar=False,
        annot=True,
        fmt=".2f",
        yticklabels= False if i != 0 else True,
    )
    axs[i].tick_params(axis='x', rotation=90)

    # Set y ticks with the index for the first plot.
    if i == 0:
        axs[i].set_yticklabels(clustered_vals.sort_values("cluster").index)
        axs[i].tick_params(axis='y', rotation=0)
    else:
        axs[i].set_yticklabels([])

if save_fig:
    plt.savefig(f"{folder}/heatmap_clustered_vals.pdf", bbox_inches="tight")

# %%
clustered_heatmap = heatmap.copy()
clustered_heatmap["cluster"] = kmeans_vals.labels_

# Plot stripplot

fig, axs = plt.subplots(1, 8, figsize=(30 * cm, 30 * cm), gridspec_kw={'wspace': 1})

# Reset the index of the heatmap DataFrame
clustered_heatmap_reset = clustered_heatmap.reset_index()

for i, col in enumerate(clustered_heatmap.columns[:-1]):
    sns.stripplot(
        data=clustered_heatmap_reset,
        y=col,
        hue="cluster",
        ax=axs[i],
        jitter=0.1,
        alpha=1,
        size=10,
        palette="tab10",
        legend=False,
    )
    axs[i].set_ylabel(col)

# Add manual legend below plots.
labels = [f"Cluster {i}" for i in range(n_clusters)]
handles = [mpl.patches.Patch(color=plt.cm.tab10(i), label=labels[i]) for i in range(n_clusters)]
axs[3].legend(handles=handles, loc='upper center', bbox_to_anchor=(1, -0.01), ncol=5)


# %%
combined_clusters = ranked_heatmap.copy()
combined_clusters["cluster"] = kmeans_vals.labels_

fig, axs = plt.subplots(1, 9, figsize=(30 * cm, 30 * cm))

# Plot heatmap with Reds

for i, col in enumerate(combined_clusters.columns[:-1]):
    sns.heatmap(
        combined_clusters.sort_values("cluster")[[col]],
        ax=axs[i],
        cmap="Reds_r",
        cbar=False,
        annot=True,
        yticklabels= False if i != 0 else True,
    )
    axs[i].tick_params(axis='x', rotation=90)

    # Set y ticks with the index for the first plot.
    if i == 0:
        axs[i].set_yticklabels(combined_clusters.sort_values("cluster").index)
        axs[i].tick_params(axis='y', rotation=0)
    else:
        axs[i].set_yticklabels([])

ax = axs.flatten()[-1]
sns.heatmap(
    combined_clusters.sort_values("cluster")[["cluster"]],
    ax=ax,
    cmap=plt.cm.tab10,
    cbar=False,
    annot=True,
    yticklabels= False,
)

if save_fig:
    plt.savefig(f"{folder}/heatmap_combined_clusters.pdf", bbox_inches="tight")

# %%
for cluster in range(n_clusters):
    print(f"Cluster {cluster}:")
    print(combined_clusters.loc[combined_clusters["cluster"] == cluster].index)

# %%
# Print the cluster centroids
cluster_centroids = pd.DataFrame(kmeans_vals.cluster_centers_ * clustered_vals.std()[:-1].values + clustered_vals.mean()[:-1].values, columns=clustered_vals.columns[:-1])
cluster_centroids.T

# %%
# Print the cluster centroids (ranks)
ranked_centroids = pd.DataFrame(columns=combined_clusters.columns[:-1])
for i in range(n_clusters):
    ranked_centroids.loc[i] = combined_clusters[combined_clusters["cluster"] == i].mean()
ranked_centroids.T

# %%
# Save results
# RANKED HEATMAP / KPI
ranked_heatmap.to_csv(f"{folder}/kpi.csv")

# CLUSTERS
clustered_vals.to_csv(f"{folder}/clustered_vals_{n_clusters}.csv")

# RANKED VALS WITH CLUSTERS
combined_clusters.to_csv(f"{folder}/combined_clusters_{n_clusters}.csv")

# PRINT CENTROIDS
cluster_centroids.to_csv(f"{folder}/centroids_{n_clusters}.csv")

# PRINT RANKED CENTROIDS
ranked_centroids.to_csv(f"{folder}/ranked_centroids_{n_clusters}.csv")


# %% [markdown]
# # Attempt at distinguishing events by power or energy event (5, energy deficit)
# ## Power events
# 
# - **Extreme power event** (Cluster 2: 1 event) Most extreme event (outlier?) by net load.
# - **Several severe power events** (Cluster 0: 9 events) Sequence of severe events with chance of recharging (either going to negative net load or close to no deficit). Strong usage of hydrogen, high net load and fuel cell usage, large deficit. Seemingly more expensive years.
# - **Severe power event** (Cluster 3: 15 events) Wind-driven. Severe power event with high usage of fuel cells, with hydrogen covering large share of deficit, strong wind anomaly. Medium difficulty of years.
# 
# ## Mixed events
# 
# - **Mixed event** (Cluster 4: 6 events) Load-driven. High load (and net load), medium duration with higher usage of fuel cells or longer duration with low H2 discharge. Difficult years?
# 
# ## Energy events
# 
# - **Energy event** (Cluster 1: 8 events) Less severe event, higher energy deficit, less hydrogen discharge or peaking, longer duration. Presumably in less severe years, as less fuel cell capacity is installed.
# 

# %% [markdown]
# # Attempt at distinguishing events by power or energy event (4, duratino)
# 


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
def compute_incidence_matrix(periods, sensitvity_periods):
    matrix = np.zeros((len(periods), len(sensitivity_periods)))
    for j, scenar in enumerate(sensitivity_periods.keys()):
        alt_periods = sensitivity_periods[scenar]
        if len(alt_periods) == 0:
            matrix[:,j] = 0
        else:
            for i, period in periods.iterrows():
                start = period.start.tz_localize(tz="UTC")
                end = period.end.tz_localize(tz="UTC")
                time_slice = pd.date_range(start, end, freq="h")
                for alt_period in alt_periods.iterrows():
                    alt_time_slice = pd.date_range(alt_period[1].start, alt_period[1].end, freq="h")
                    if len(set(time_slice).intersection(set(alt_time_slice))) > 0:
                        matrix[i,j] = 1
                        break
    return matrix

def plot_incidence_matrix(m, sensitivity_periods, cmap=mpl.colors.ListedColormap(["red", "green"]), ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    im = ax.imshow(m, cmap=cmap)
    ax.set_xlabel("Sensitivity scenarios")
    ax.set_ylabel("System-defining events")
    ax.set_xticks(np.arange(len(sensitivity_periods)))
    ax.set_xticklabels([f"{scenar[0]}_{scenar[1]}_{scenar[2]}" for scenar in sensitivity_periods.keys()], rotation=90)
    return im


# %%
# Compute the number of system-defining events per sensitivity scenario
n_events = {cost: {
    scenar: len(sens_periods[cost][scenar]) for scenar in sens_periods[cost].keys()} for cost in cost_thresholds}
pd.concat({cost: pd.DataFrame(n_events[cost], index=["n_events"]).T for cost in cost_thresholds}, axis=1)

# %%
for cost in cost_thresholds:
    print(f"Cost threshold: {cost}")
    sensitivity_periods = sens_periods[cost]
    incidence_matrix = compute_incidence_matrix(periods, sensitivity_periods)
    fig, ax = plt.subplots(figsize=(18 * cm, 18 * cm))
    plot_incidence_matrix(incidence_matrix, sensitivity_periods, ax=ax)
    plt.show()

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

# %%
# # Standard vs c1.0, EQ0.7
# plot_duals(
#         periods, 
#         years, 
#         total_costs, 
#         total_costs, 
#         mpl.colors.LogNorm(vmin=0.1, vmax=10000, clip=True),
#         mpl.colors.LogNorm(vmin=0.1, vmax=10000, clip=True), 
#         "Oranges",
#         "Oranges",
#         "Hourly electricity cost (million EUR)",
#         "Hourly electricity cost (million EUR)",
#         [1, 10, 100, 1000, 10000],
#         [1, 10, 100, 1000, 10000],
#         save_fig=False,
#         path_str = None,
#         alt_periods = sens_periods[100][(90, 'c1.0', 'Co2L0.0-EQ0.7')]
#     )

# %%
# # Standard vs c1.0, EQ0.9
# plot_duals(
#         periods, 
#         years, 
#         total_costs, 
#         total_costs, 
#         mpl.colors.LogNorm(vmin=0.1, vmax=10000, clip=True),
#         mpl.colors.LogNorm(vmin=0.1, vmax=10000, clip=True), 
#         "Oranges",
#         "Oranges",
#         "Hourly electricity cost (million EUR)",
#         "Hourly electricity cost (million EUR)",
#         [1, 10, 100, 1000, 10000],
#         [1, 10, 100, 1000, 10000],
#         save_fig=False,
#         path_str = None,
#         alt_periods = sens_periods[100][(90, 'c1.0', 'Co2L0.0-EQ0.9')]
#     )

# %%
# # Standard vs c1.25, EQ0.7
# plot_duals(
#         periods, 
#         years, 
#         total_costs, 
#         total_costs, 
#         mpl.colors.LogNorm(vmin=0.1, vmax=10000, clip=True),
#         mpl.colors.LogNorm(vmin=0.1, vmax=10000, clip=True), 
#         "Oranges",
#         "Oranges",
#         "Hourly electricity cost (million EUR)",
#         "Hourly electricity cost (million EUR)",
#         [1, 10, 100, 1000, 10000],
#         [1, 10, 100, 1000, 10000],
#         save_fig=False,
#         path_str = None,
#         alt_periods = sens_periods[100][(90, 'c1.25', 'Co2L0.0-EQ0.7')]
#     )

# %%
# # Standard vs c1.25, EQ0.9
# plot_duals(
#         periods, 
#         years, 
#         total_costs, 
#         total_costs, 
#         mpl.colors.LogNorm(vmin=0.1, vmax=10000, clip=True),
#         mpl.colors.LogNorm(vmin=0.1, vmax=10000, clip=True), 
#         "Oranges",
#         "Oranges",
#         "Hourly electricity cost (million EUR)",
#         "Hourly electricity cost (million EUR)",
#         [1, 10, 100, 1000, 10000],
#         [1, 10, 100, 1000, 10000],
#         save_fig=False,
#         path_str = None,
#         alt_periods = sens_periods[100][(90, 'c1.0', 'Co2L0.0-EQ0.9')]
#     )

# %%
# # Standard vs c2
# plot_duals(
#         periods, 
#         years, 
#         total_costs, 
#         total_costs, 
#         mpl.colors.LogNorm(vmin=0.1, vmax=10000, clip=True),
#         mpl.colors.LogNorm(vmin=0.1, vmax=10000, clip=True), 
#         "Oranges",
#         "Oranges",
#         "Hourly electricity cost (million EUR)",
#         "Hourly electricity cost (million EUR)",
#         [1, 10, 100, 1000, 10000],
#         [1, 10, 100, 1000, 10000],
#         save_fig=False,
#         path_str = None,
#         alt_periods = sens_periods[100][(90, 'c2', 'Co2L0.0')]
#     )

# %%
# # Standard vs c2, EQ0.7
# plot_duals(
#         periods, 
#         years, 
#         total_costs, 
#         total_costs, 
#         mpl.colors.LogNorm(vmin=0.1, vmax=10000, clip=True),
#         mpl.colors.LogNorm(vmin=0.1, vmax=10000, clip=True), 
#         "Oranges",
#         "Oranges",
#         "Hourly electricity cost (million EUR)",
#         "Hourly electricity cost (million EUR)",
#         [1, 10, 100, 1000, 10000],
#         [1, 10, 100, 1000, 10000],
#         save_fig=False,
#         path_str = None,
#         alt_periods = sens_periods[100][(90, 'c2', 'Co2L0.0-EQ0.7')]
#     )

# %%
# # Standard vs c2, EQ0.9
# plot_duals(
#         periods, 
#         years, 
#         total_costs, 
#         total_costs, 
#         mpl.colors.LogNorm(vmin=0.1, vmax=10000, clip=True),
#         mpl.colors.LogNorm(vmin=0.1, vmax=10000, clip=True), 
#         "Oranges",
#         "Oranges",
#         "Hourly electricity cost (million EUR)",
#         "Hourly electricity cost (million EUR)",
#         [1, 10, 100, 1000, 10000],
#         [1, 10, 100, 1000, 10000],
#         save_fig=False,
#         path_str = None,
#         alt_periods = sens_periods[100][(90, 'c2', 'Co2L0.0-EQ0.9')]
#     )

# %% [markdown]
# # 50 bn EUR

# %%
# # Standard vs c1.0, 5% limit
# plot_duals(
#         periods, 
#         years, 
#         total_costs, 
#         total_costs, 
#         mpl.colors.LogNorm(vmin=0.1, vmax=10000, clip=True),
#         mpl.colors.LogNorm(vmin=0.1, vmax=10000, clip=True), 
#         "Oranges",
#         "Oranges",
#         "Hourly electricity cost (million EUR)",
#         "Hourly electricity cost (million EUR)",
#         [1, 10, 100, 1000, 10000],
#         [1, 10, 100, 1000, 10000],
#         save_fig=False,
#         path_str = None,
#         alt_periods = sens_periods[50][(90, 'c1.0', 'Co2L0.05')]
#     )

# %%
# # Standard vs c1.0, 5% limit, 0.9EQ
# plot_duals(
#         periods, 
#         years, 
#         total_costs, 
#         total_costs, 
#         mpl.colors.LogNorm(vmin=0.1, vmax=10000, clip=True),
#         mpl.colors.LogNorm(vmin=0.1, vmax=10000, clip=True), 
#         "Oranges",
#         "Oranges",
#         "Hourly electricity cost (million EUR)",
#         "Hourly electricity cost (million EUR)",
#         [1, 10, 100, 1000, 10000],
#         [1, 10, 100, 1000, 10000],
#         save_fig=False,
#         path_str = None,
#         alt_periods = sens_periods[50][(90, 'c1.0', 'Co2L0.05-EQ0.9')]
#     )

# %% [markdown]
# # 30 bn EUR

# %%
# # Standard vs c1.25, 5% limit
# plot_duals(
#         periods, 
#         years, 
#         total_costs, 
#         total_costs, 
#         mpl.colors.LogNorm(vmin=0.1, vmax=10000, clip=True),
#         mpl.colors.LogNorm(vmin=0.1, vmax=10000, clip=True), 
#         "Oranges",
#         "Oranges",
#         "Hourly electricity cost (million EUR)",
#         "Hourly electricity cost (million EUR)",
#         [1, 10, 100, 1000, 10000],
#         [1, 10, 100, 1000, 10000],
#         save_fig=False,
#         path_str = None,
#         alt_periods = sens_periods[30][(90, 'c1.25', 'Co2L0.05')]
#     )

# %%
# # Standard vs c1.0, 5% limit, EQ0.9
# plot_duals(
#         periods, 
#         years, 
#         total_costs, 
#         total_costs, 
#         mpl.colors.LogNorm(vmin=0.1, vmax=10000, clip=True),
#         mpl.colors.LogNorm(vmin=0.1, vmax=10000, clip=True), 
#         "Oranges",
#         "Oranges",
#         "Hourly electricity cost (million EUR)",
#         "Hourly electricity cost (million EUR)",
#         [1, 10, 100, 1000, 10000],
#         [1, 10, 100, 1000, 10000],
#         save_fig=False,
#         path_str = None,
#         alt_periods = sens_periods[30][(90, 'c1.0', 'Co2L0.05-EQ0.9')]
#     )



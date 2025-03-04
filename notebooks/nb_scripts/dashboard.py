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
from scipy.spatial import ConvexHull

from matplotlib.ticker import (AutoMinorLocator, MultipleLocator, FormatStrFormatter,AutoMinorLocator)
import matplotlib.dates as mdates
from matplotlib.path import Path
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors

import geopandas as gpd
import cartopy
import cartopy.crs as ccrs

from typing import NamedTuple, Optional

from _notebook_utilities import *
from _plot_affected_areas import *
from _dashboard import *

import logging

# Suppress warnings and info messages from 'pypsa.io'
logging.getLogger("pypsa.io").setLevel(logging.ERROR)

cm = 1 / 2.54  # centimeters in inches

# %% [markdown]
# # Initiation

# %%
regenerate_data = False
overwrite_data = False
config_name = "stressful-weather"
config, scenario_def, years, opt_networks = load_opt_networks(config_name, load_networks=False)
periods = load_periods(config)
projection = ccrs.PlateCarree()
cluster_nr = 4

# In order to regenerate data, run `generate_data_for_analysis.py`.

# %%
# Load onshore and offshore regions for shapefile.
onshore_regions = gpd.read_file(f"../resources/{config_name}/weather_year_1941/regions_onshore_base_s_90.geojson")
# Load one network for reference and the layout.
n = pypsa.Network("../results/stressful-weather/weather_year_1941/networks/base_s_90_elec_lc1.25_Co2L.nc")

# Load all data we might need that is pre-generated in `generate_data_for_analysis.py`.
folder = f"./processing_data/{config_name}"
# Load: total load, winter load
total_load = pd.read_csv(f"{folder}/total_load.csv", index_col=0, parse_dates=True)
winter_load = pd.read_csv(f"{folder}/winter_load.csv", index_col=0)
# Annual CFS for solar and wind
annual_cfs = pd.read_csv(f"{folder}/annual_cfs.csv", index_col=0)
# Costs: total electricity costs
total_costs_df = pd.read_csv(f"{folder}/total_costs.csv", index_col=[0,1])
## SDEs
stats_periods = pd.read_csv(f"{folder}/stats_periods.csv", index_col=0)
gen_stacks = pd.read_csv(f"{folder}/gen_stacks.csv", index_col=0, parse_dates=True)
# Flexibility usage and capacities
nodal_flex_p = pd.read_csv(f"{folder}/nodal_flex_p.csv", index_col=[0,1]) 
nodal_flex_u = xr.open_dataset(f"processing_data/{config_name}/nodal_flex_u.nc")
fc_flex = nodal_flex_u["H2 fuel cell"].to_pandas().T
fc_flex.index = total_load.index
system_flex_p = (nodal_flex_p.unstack().sum(axis="rows") / 1e3).round(1)
flex_caps = system_flex_p[["battery discharger", "H2 fuel cell"]].unstack(level=0)
# Ranked years: NOTE that this refers to highest load shedding, not over the annual sum
ranked_years = pd.read_csv(f"{folder}/ranked_years.csv", index_col=0)
total_costs = {}
for year in years:
    df = total_costs_df.loc[year]
    df.index = pd.to_datetime(df.index)
    total_costs[year] = df["0"]

# Generate a dataframe with all necessary annual values: 
annual_values = collect_annual_values(ranked_years, annual_cfs, winter_load, total_costs, flex_caps, years)

# Load clusters
clusters = pd.read_csv(f"clustering/{config_name}/clustered_vals_{cluster_nr}.csv", index_col=0)["cluster"]
stats_periods["cluster"] = clusters

# Load hulls
hulls_coll = load_hull_data(config_name, periods, techs = ["wind_anom", "load_anom", "price"], thres = [0.75, 0.9, 0.9])
hulls_markers_names = ["Wind anomaly (-)", "Load anomaly (+)", "Shadow price (+)", "Fuel cell discharge (+)"]


# %%
annual_values

# %%
# Plot dashboard.
for event_nr in periods.index:
    plot_dashboard(
        config_name = "stressful-weather",
        event_nr = event_nr,
        stats_periods = stats_periods,
        annual_values = annual_values,
        kpis = ["highest_net_load", "avg_net_load", "wind_anom", "avg_rel_load", "max_fc_discharge", "duration", "normed_price_std"],
        kpi_names = ["Peak net load [GW]","Avg. net load [GW]","Wind CF anomaly","Avg. rel. load","Max. Fuel cell discharge [GW]","Duration [h]","Regional price imbalance"],
        hulls_coll=hulls_coll,
        hulls_markers_names=hulls_markers_names,
        fc_flex=fc_flex,
        onshore_regions=onshore_regions,
        n = n,
        projection = projection,
        categories = ["difficulty", "weather", "cost", "other"],
        cat_names = ["Difficulty rank", "Weather", "Costs of SDE", "Flexibility"],
        label_names = {"difficulty": ["Total system costs", "LS design", "LS operation"], "weather":["Solar CF", "Wind CF", "Winter load"],
        "cost": ["Share of total costs", "Recovery of FC inv.", "Recovery of battery inv."],
        "other": ["Battery discharger \n capacity[GW]", "Fuel cell capacity \n [GW]"]},
        gen_stacks = gen_stacks,
        time_window = pd.Timedelta("7d"),
        total_load = total_load,
        freq = "3H",
        save = False
    )



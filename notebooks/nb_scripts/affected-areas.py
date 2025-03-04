# %%
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="facecolor will have no effect as it has been defined as",
)

import os
import sys
from multiprocessing import Pool

import yaml
import pypsa
import datetime as dt

import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["figure.dpi"] = 150
# Generate a matplotlib patch for the convex hull.
from matplotlib.patches import Polygon
from matplotlib.patches import Patch


import pandas as pd
import geopandas as gpd
import numpy as np
import cartopy
import cartopy.crs as ccrs
import seaborn as sns
import xarray as xr

from sklearn.cluster import KMeans
import networkx as nx
from scipy.spatial import ConvexHull

import geopy.distance

from _notebook_utilities import *
from _plot_affected_areas import *

# %%
config_name = "stressful-weather"
folder = f"./processing_data/{config_name}"

config, scenario_def, years, opt_networks = load_opt_networks(config_name, load_networks=False)

periods = load_periods(config)
prices = pd.read_csv(f"{folder}/all_prices.csv", index_col=0, parse_dates=True)

# %%
# Load onshore and offshore regions for shapefile.
onshore_regions = gpd.read_file("../resources/stressful-weather/weather_year_1941/regions_onshore_base_s_90.geojson")
offshore_regions = gpd.read_file("../resources/stressful-weather/weather_year_1941/regions_offshore_base_s_90.geojson")

# Load one network for reference and the layout.
n = pypsa.Network("../results/stressful-weather/weather_year_1941/networks/base_s_90_elec_lc1.25_Co2L.nc")


# Load the means, anomalies.
load_means = pd.read_csv("../results/means/load_1941-2020_100bn_12-336h_90_elec_lc1.25_Co2L.csv", index_col=0, parse_dates=True)
solar_means = pd.read_csv("../results/means/solar_1941-2020_100bn_12-336h_90_elec_lc1.25_Co2L.csv", index_col=0, parse_dates=True)
wind_means = pd.read_csv("../results/means/wind_1941-2020_100bn_12-336h_90_elec_lc1.25_Co2L.csv", index_col=0, parse_dates=True)

# Load costs
total_costs_df = pd.read_csv(f"processing_data/{config_name}/total_costs.csv", index_col=[0,1])
total_costs = {}
for year in years:
    df = total_costs_df.loc[year]
    df.index = pd.to_datetime(df.index)


# Load flexibility indicators.
# Usage of flexibility indicators.
nodal_flex_u = xr.open_dataset(f"processing_data/{config_name}/nodal_flex_u.nc")
# Extracted for periods.
# nodal_flex_periods = pd.read_csv(f"processing_data/nodal_flex_periods.csv", index_col=[0,1], parse_dates=True)
# Seasonality of nodal flexibility.
# nodal_flex_seasonality = pd.read_csv(f"processing_data/nodal_flex_seasonality.csv", index_col=[0,1])

# Capacities.
nodal_flex_p = pd.read_csv(f"processing_data/{config_name}/nodal_flex_p.csv", index_col=[0,1])

# Anomalies during periods and peak hours in flexibility usage.
nodal_flex_anomaly_periods = pd.read_csv(f"processing_data/{config_name}/nodal_anomaly_flex_u.csv", index_col=[0,1], parse_dates=True)
nodal_flex_anomaly_peak = pd.read_csv(f"processing_data/{config_name}/nodal_peak_anomaly_flex_u.csv", index_col=[0,1], parse_dates=True)

# Capacity factors and capacities.
wind_cf = xr.open_dataset(f"processing_data/{config_name}/wind_cf.nc").to_dataframe()
solar_cf = xr.open_dataset(f"processing_data/{config_name}/solar_cf.nc").to_dataframe()
# wind_caps = pd.read_csv(f"processing_data/{config_name}/wind_caps.csv", index_col=0)
# solar_caps = pd.read_csv(f"processing_data/{config_name}/solar_caps.csv", index_col=0)

# Load laod.
total_load = pd.read_csv(f"processing_data/{config_name}/total_load.csv", index_col=0, parse_dates=True)
nodal_load = pd.read_csv(f"processing_data/{config_name}/nodal_load.csv", index_col=0, parse_dates=True)

# Load state of charge
avg_soc = pd.read_csv(f"processing_data/{config_name}/avg_soc.csv", index_col=0, parse_dates=True)
state_of_charge = pd.read_csv(f"processing_data/{config_name}/state_of_charge.csv", index_col=0, parse_dates=True)


# %%
# Figure settings
projection = ccrs.PlateCarree()

# Anomalies in "weather": wind, solar, load
wind_norm = mpl.colors.Normalize(vmin=-0.3, vmax=0.3)
solar_norm = mpl.colors.Normalize(vmin=-0.1, vmax=0.1)
load_norm = mpl.colors.Normalize(vmin=-0.2, vmax=0.2)

# System variables
# Prices: nominal
# prices_norm = mpl.colors.Normalize(vmin=0, vmax=3000)
# Prices: logarithmic
prices_norm = mpl.colors.LogNorm(vmin=100, vmax=3000)

# # Storage anomalies
# storage_norm = mpl.colors.Normalize(vmin=-0.3, vmax=0.3)

# Flexibility usage: no
# transmission_norm = mpl.colors.Normalize(vmin=0, vmax=1)
# dispatch_norm = mpl.colors.Normalize(vmin=0, vmax=1)
# discharge_norm = mpl.colors.Normalize(vmin=0, vmax=1)
fc_norm = mpl.colors.Normalize(vmin=0, vmax=1)


# %%
grid_wind(
    config_name = config_name,
    periods = periods,
    df = wind_cf,
    cluster_sense = "min",
    tech = "wind",
    norm = wind_norm,
    regions = onshore_regions,
    offshore_regions = offshore_regions,
    projection = projection,
    n = n,
    cluster_nb = 3,
    threshold = 0.75,
    averages = wind_means,
    save = False,
    cmap = "coolwarm"
)

# %%
grid_maps(
    config_name,
    periods,
    solar_cf,
    "min",
    "solar",
    solar_norm,
    onshore_regions,
    projection,
    n,
    3,
    0.7,
    use_anomalies = True,
    averages = solar_means,
    save = True,
    cmap = "coolwarm",
)

# %%
grid_maps(
    config_name,
    periods,
    nodal_load,
    "max",
    "load",
    load_norm,
    onshore_regions,
    projection,
    n,
    2,
    0.9,
    use_anomalies = True,
    averages = load_means,
    save = True,
    cmap = "coolwarm",
)

# %%
prices_norm = mpl.colors.LogNorm(vmin=500, vmax=5000)
grid_maps(
    config_name,
    periods,
    prices,
    "max",
    "price",
    prices_norm,
    onshore_regions,
    projection,
    n,
    2,
    0.9,
    use_anomalies = False,
    averages = None,
    save = False,
    cmap = "Reds",
)

# %%
fc_flex = nodal_flex_u["H2 fuel cell"].to_pandas().T
fc_flex.index = total_load.index

grid_maps(
    config_name,
    periods,
    fc_flex,
    "max",
    "fuel_cells",
    fc_norm,
    onshore_regions,
    projection,
    n,
    2,
    0.85,
    use_anomalies = False,
    averages = None,
    save = True,
    cmap = "Purples",
)

# %%
techs = ["wind_anom", "load_anom", "price"
#, "fuel_cells"
]
pretty_names = ["Wind anomaly (-)", "Load anomaly (+)", "Shadow price (+)", "Fuel cell discharge (+)"]
thres = [0.75, 0.9, 0.9, 0.85]

wind_hulls, load_hulls, prices_hulls, fc_hulls = [], [], [], []
hulls_collection = [wind_hulls, load_hulls, prices_hulls, fc_hulls]
[wind_hulls, load_hulls, prices_hulls, fc_hulls]
for tech, t, h in zip(techs, thres, hulls_collection):
    h.extend([ConvexHull(pd.read_csv(f"processing_data/{config_name}/maps/{tech}/hull_{t}_event{i}.csv", index_col=0)) for i in range(len(periods))])
hulls_coll = {k: v for k, v in zip(techs, [wind_hulls, load_hulls, prices_hulls, fc_hulls])}










# %%
grid_affected_areas(
    config_name,
    periods,
    hulls_coll,
    techs,
    pretty_names,
    ["#235ebc", "#dd2e23", "green", '#c251ae'],
    fc_flex,
    "fuel_cells",
    fc_norm,
    "Purples",
    onshore_regions,
    n,
    projection,
    save=False,
)

# %%




# %%
# Initiation

import pypsa 
import datetime as dt 

import yaml

import matplotlib.pyplot as plt
import matplotlib as mpl

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

# %%
# Load all data we might need that is pre-generated in `generate_data_for_analysis.py`.
folder = f"./processing_data/{config_name}"

# Objective values
opt_objs = pd.read_csv(f"{folder}/opt_objs.csv", index_col=0)
reindex_opt_objs = opt_objs.copy().sum(axis="columns") 
reindex_opt_objs.index=years

# Means
avg_load = pd.read_csv(f"../results/means/load_1941-2020_100bn_12-336h_90_elec_lc1.25_Co2L.csv", index_col=0, parse_dates=True)
avg_wind = pd.read_csv(f"../results/means/wind_1941-2020_100bn_12-336h_90_elec_lc1.25_Co2L.csv", index_col=0, parse_dates=True)
avg_solar = pd.read_csv(f"../results/means/solar_1941-2020_100bn_12-336h_90_elec_lc1.25_Co2L.csv", index_col=0, parse_dates=True)

# Load: net load, total load, winter load
net_load = pd.read_csv(f"{folder}/net_load.csv", index_col=0, parse_dates=True)
total_load = pd.read_csv(f"{folder}/total_load.csv", index_col=0, parse_dates=True)

# Costs: nodal prices, total electricity/storage/fuel cell costs
all_prices = pd.read_csv(f"{folder}/all_prices.csv", index_col=0, parse_dates=True)
total_costs_df = pd.read_csv(f"{folder}/total_costs.csv", index_col=[0,1])
total_costs_df.index = total_costs_df.index.droplevel(0)
total_costs_df.index = pd.to_datetime(total_costs_df.index)
total_storage_costs_df = pd.read_csv(f"{folder}/total_storage_costs.csv", index_col=[0,1])
total_fc_costs_df = pd.read_csv(f"{folder}/total_fc_costs.csv", index_col=[0,1])

# Storage: storage capacities, storage levels, average storage levels
# s_caps = pd.read_csv(f"{folder}/s_caps.csv", index_col=0)
# su_soc = pd.read_csv(f"{folder}/state_of_charge.csv", index_col=0, parse_dates=True)
# avg_soc = pd.read_csv(f"{folder}/avg_soc.csv", index_col=0, parse_dates=True)

# Transmission:


# Capacity (factors) for wind and solar; wind distribution in the winter
wind_caps = pd.read_csv(f"{folder}/wind_caps.csv", index_col=0)
solar_caps = pd.read_csv(f"{folder}/solar_caps.csv", index_col=0)
wind_cf = xr.open_dataset(f"processing_data/{config_name}/wind_cf.nc").to_dataframe()
solar_cf = xr.open_dataset(f"processing_data/{config_name}/solar_cf.nc").to_dataframe()
wind_distr_df = pd.read_csv(f"processing_data/{config_name}/wind_distr.csv", index_col=[0,1],)


## SDEs
# Stats for storage behaviour
stores_periods = pd.read_csv(f"{folder}/stores_periods.csv", index_col=0)

# Stats for clustering: net load peak hour, highest net load, avg net load, energy deficit, h2 discharge, max fc discharge, avg rel load, wind cf, wind anom, annual cost
stats_periods = pd.read_csv(f"{folder}/stats_periods.csv", index_col=0)





# %%

## FLEXIBILITY
# System: detailed
# all_flex_detailed = pd.read_csv(f"{folder}/all_flex_detailed.csv", index_col=0, parse_dates=True)
# avg_flex_detailed = pd.read_csv(f"{folder}/avg_flex_detailed.csv", index_col=0, parse_dates=True)
# periods_flex_detailed = pd.read_csv(f"{folder}/periods_flex_detailed.csv", index_col=0, parse_dates=True)
# periods_anomaly_flex_detailed = pd.read_csv(f"{folder}/periods_anomaly_flex_detailed.csv", index_col=0, parse_dates=True)
# periods_peak_flex_detailed = pd.read_csv(f"{folder}/periods_peak_flex_detailed.csv", index_col=0, parse_dates=True)
# periods_peak_anomaly_flex_detailed = pd.read_csv(f"{folder}/periods_peak_anomaly_flex_detailed.csv", index_col=0, parse_dates=True)
# System: coarse
#all_flex_coarse = pd.read_csv(f"{folder}/all_flex_coarse.csv", index_col=0, parse_dates=True)
#avg_flex_coarse = pd.read_csv(f"{folder}/avg_flex_coarse.csv", index_col=0, parse_dates=True)
#periods_flex_coarse = pd.read_csv(f"{folder}/periods_flex_coarse.csv", index_col=0, parse_dates=True)
#periods_anomaly_flex_coarse = pd.read_csv(f"{folder}/periods_anomaly_flex_coarse.csv", index_col=0, parse_dates=True)
#periods_peak_flex_coarse = pd.read_csv(f"{folder}/periods_peak_flex_coarse.csv", index_col=0, parse_dates=True)
#periods_peak_anomaly_flex_coarse = pd.read_csv(f"{folder}/periods_peak_anomaly_flex_coarse.csv", index_col=0, parse_dates=True)



# # Nodal
# nodal_flex_p = pd.read_csv(f"{folder}/nodal_flex_p.csv", index_col=[0,1])
# nodal_seasonality = pd.read_csv(f"{folder}/nodal_seasonality.csv", index_col=0, parse_dates=True)
# nodal_flex_periods = pd.read_csv(f"{folder}/nodal_flex_periods.csv", index_col=0, parse_dates=True)
# nodal_flex_anomaly_periods = pd.read_csv(f"{folder}/nodal_flex_anomaly_periods.csv", index_col=0, parse_dates=True)
# nodal_peak_anomaly_flex = pd.read_csv(f"{folder}/nodal_peak_anomaly_flex.csv", index_col=0, parse_dates=True)


# ## SYSTEM VALUES
# all_system_anomaly = pd.read_csv(f"{folder}/all_system_anomaly.csv", index_col=0, parse_dates=True)
# all_used_flexibility = pd.read_csv(f"{folder}/all_used_flexibility.csv", index_col=0, parse_dates=True)
# all_flex_anomaly = pd.read_csv(f"{folder}/all_flex_anomaly.csv", index_col=0, parse_dates=True)

# %%
total_costs = {}
for year in years:
    df = total_costs_df.loc[f"{year}"]
    df.index = pd.to_datetime(df.index)
    total_costs[year] = df["0"]

# %%
if regenerate_data:
    # Find the 20 worst hours of net load that are not in any period.

    max_nr = 20

    worst_found_hours = pd.DataFrame(columns=["Net load", "Event"])
    worst_unid_hours = pd.DataFrame(columns=["Net load"])

    found_nr = 0

    worst_hours = net_load.sort_values("Net load", ascending=False)

    for hour in worst_hours.index:
        in_period = False
        for i, period in periods.iterrows():
            if period.start <= hour <= period.end:
                worst_found_hours.loc[hour] = [net_load.loc[hour, "Net load"], i]
                in_period = True
                break
        if not in_period:
            worst_unid_hours.loc[hour, "Net load"] = net_load.loc[hour, "Net load"]
            found_nr += 1
            if found_nr >= max_nr:
                break

    df_help = worst_unid_hours.copy()
    for i, hour in enumerate(df_help.index):
        other_hours = [h for h in df_help.index[i+1:]]
        for oh in other_hours:
            if hour.date() == oh.date():
                worst_unid_hours.drop(oh, inplace=True)

    worst_unid_hours /= 1e3
    worst_found_hours["Net load"] /= 1e3

    if overwrite_data:
        worst_unid_hours.to_csv(f"{folder}/weather_filtering/worst_unid_hours.csv")
    else:
        worst_unid_hours = pd.read_csv(f"{folder}/weather_filtering/worst_unid_hours.csv", index_col=0, parse_dates=True)



# %%
# Compute duration of each period
for i, period in periods.iterrows():
    stats_periods.loc[i, "duration"] = (pd.Timestamp(period.end) - pd.Timestamp(period.start)).total_seconds() / 3600

# %%
duration_periods = stats_periods.sort_values("duration", ascending=False)

# %%
def stats_sde(
        periods,
        stores_periods,
        net_load,
        total_load,
        avg_load,
        avg_wind,
        wind_cf,
        wind_caps,
        reindex_opt_objs):
    '''
    Compute the statistics for the system-defining events.'''
    stats_periods = periods.copy()
    for i in stats_periods.index:
        stats_periods.loc[i, "net_load_peak_hour"] = net_load.loc[stats_periods.loc[i, "peak_hour"], "Net load"]/ 1e3
        stats_periods.loc[i, "highest_net_load"] = net_load.loc[stats_periods.loc[i, "start"]:stats_periods.loc[i, "end"], "Net load"].max() / 1e3
        stats_periods.loc[i, "avg_net_load"] = net_load.loc[stats_periods.loc[i, "start"]:stats_periods.loc[i, "end"], "Net load"].mean()/ 1e3
        stats_periods.loc[i, "energy_deficit"] = net_load.loc[stats_periods.loc[i, "start"]:stats_periods.loc[i, "end"], "Net load"].sum()/ 1e6
        stats_periods.loc[i, "h2_discharge"] = stores_periods.loc[stores_periods.index[i], "discharge"].sum()
        stats_periods.loc[i, "max_fc_discharge"] = stores_periods.loc[stores_periods.index[i], "max_fc_discharge"].sum()
        start, end = stats_periods.loc[i, "start"], stats_periods.loc[i, "end"]
        year_s = 1942 if start.month < 7 else 1941
        year_e = 1942 if end.month < 7 else 1941
        shifted_start = pd.Timestamp(f"{year_s}-{str(start.month).zfill(2)}-{str(start.day).zfill(2)} {str(start.hour).zfill(2)}:00:00")
        shifted_end = pd.Timestamp(f"{year_e}-{str(end.month).zfill(2)}-{str(end.day).zfill(2)} {str(end.hour).zfill(2)}:00:00")
        helper_df = total_load.loc[start:end, "0"].values / avg_load.loc[shifted_start:shifted_end].sum(axis=1)
        helper_df = pd.DataFrame(helper_df, index = pd.date_range(start, end, freq="h"))
        stats_periods.loc[i, "avg_rel_load"] = np.mean((total_load.loc[start:end, "0"])/(avg_load.loc[shifted_start:shifted_end].sum(axis=1).values))
        n_year = str(get_net_year(start))
        stats_periods.loc[i, "wind_cf"] = ((wind_cf @ wind_caps[n_year])/wind_caps[n_year].sum()).loc[start:end].mean()
        wind_anom = stats_periods.loc[i, "wind_cf"] - (
            (avg_wind @ wind_caps[n_year])/wind_caps[n_year].sum()
        ).loc[shifted_start:shifted_end].mean()
        stats_periods.loc[i, "wind_anom"] = wind_anom       
        stats_periods = stats_periods.round(3)
        stats_periods[["net_load_peak_hour", "highest_net_load", "avg_net_load", ]] = stats_periods[["net_load_peak_hour", "highest_net_load", "avg_net_load"]].round(0)
        net_y =  int(get_net_year(periods.loc[i, "start"]))
        stats_periods.loc[i, "annual_cost"] = int(reindex_opt_objs.loc[net_y] / 1e9)
    return stats_periods

# %%
if overwrite_data:
    # Some pre-computations.

    # Wind anomaly
    total_wind_anomaly = pd.DataFrame(index = wind_cf.index, columns = ["Wind anomaly"]).astype(float)
    df_helper = pd.concat(80*[avg_wind])
    df_helper.index = wind_cf.index
    for year in years:
        total_wind_anomaly.loc[f"{year}-07-01":f"{year+1}-06-30", "Wind anomaly"] = ((wind_cf.loc[f"{year}-07-01":f"{year+1}-06-30"] @ wind_caps[f"{year}"]) - (df_helper.loc[f"{year}-07-01":f"{year+1}-06-30"] @ wind_caps[f"{year}"]))/wind_caps[f"{year}"].sum()

    # Relative load
    total_rel_load = pd.DataFrame(index = total_load.index, columns = ["Relative load"]).astype(float)
    df_helper = pd.concat(80*[avg_load.sum(axis=1)])
    df_helper.index = total_load.index
    for year in years:
        total_rel_load.loc[f"{year}-07-01":f"{year+1}-06-30", "Relative load"] = total_load.loc[f"{year}-07-01":f"{year+1}-06-30", "0"] / df_helper.loc[f"{year}-07-01":f"{year+1}-06-30"]

    
    ## FILTER BY WEATHER
    
    
    all_three = pd.DataFrame(index=pd.IntervalIndex.from_tuples([], closed="both", dtype="interval[datetime64[ns], both]"), columns=["Wind anomaly", "Avg. rel. load", "Avg. net load", "Max net load", "Costs"]).astype(float)
    # not_wind = pd.DataFrame(index=pd.IntervalIndex.from_tuples([], closed="both", dtype="interval[datetime64[ns], both]"), columns=["Relative load", "Net load"]).astype(float)
    # not_avg_rel = pd.DataFrame(index=pd.IntervalIndex.from_tuples([], closed="both", dtype="interval[datetime64[ns], both]"), columns=["Wind anomaly", "Net load"]).astype(float)
    # not_avg_net = pd.DataFrame(index=pd.IntervalIndex.from_tuples([], closed="both", dtype="interval[datetime64[ns], both]"), columns=["Wind anomaly", "Relative load"]).astype(float)


    for i, period in duration_periods.iterrows():
        start = pd.Timestamp(period.start)
        end = pd.Timestamp(period.end)
        dur = int(period.duration)
        anl = period.avg_net_load * 1e3 # was before in GW
        wanom = period.wind_anom
        arl = period.avg_rel_load
        event_nr = period.name

        intervals = pd.IntervalIndex.from_arrays(
                left=total_wind_anomaly.index[:-dur], right=total_wind_anomaly.index[dur:], closed="both"
            )  
        
        wind_anomaly = total_wind_anomaly.rolling(dur).mean().iloc[dur:]
        average_relative_load = total_rel_load.rolling(dur).mean().iloc[dur:]
        average_net_load = net_load.rolling(dur).mean().iloc[dur:]
        wind_anomaly.index = intervals
        average_relative_load.index = intervals
        average_net_load.index = intervals

        # Remove all overlaps with existing SDEs.
        for i, per in periods.iterrows():
            per_start = pd.Timestamp(per.start)
            per_end = pd.Timestamp(per.end)

            overlapping_intervals = pd.IntervalIndex.from_arrays(
                    left=total_wind_anomaly.loc[per_start-pd.Timedelta(hours=dur):per_end + pd.Timedelta(hours=dur)].index[:-dur], right=total_wind_anomaly.loc[per_start - pd.Timedelta(hours=dur):per_end + pd.Timedelta(hours=dur)].index[dur:], closed="both"
                )
            intervals = intervals.difference(overlapping_intervals)

        wind_anomaly = wind_anomaly.loc[intervals]
        average_relative_load = average_relative_load.loc[intervals]
        average_net_load = average_net_load.loc[intervals]

        
        wind_anomaly = wind_anomaly.loc[wind_anomaly.index.length <= pd.Timedelta(hours=dur)]
        average_relative_load = average_relative_load.loc[average_relative_load.index.length <= pd.Timedelta(hours=dur)]
        average_net_load = average_net_load.loc[average_net_load.index.length <= pd.Timedelta(hours=dur)]


        # Filter out periods that do not meet the criteria.
        wind_anomaly = wind_anomaly.loc[wind_anomaly["Wind anomaly"] < wanom]
        average_relative_load = average_relative_load.loc[average_relative_load["Relative load"] > arl]
        average_net_load = average_net_load.loc[average_net_load["Net load"] > anl]

        # Find the intersection of the three.
        three = (wind_anomaly.index.intersection(average_relative_load.index)).intersection(average_net_load.index)
        # not_wind_anomaly = average_relative_load.index.intersection(average_net_load.index)
        # not_average_relative_load = wind_anomaly.index.intersection(average_net_load.index)
        # not_average_net_load = wind_anomaly.index.intersection(average_relative_load.index)

        # Filter out periods that might have been identified before.
        candidates_three = wind_anomaly.copy()
        candidates_three = candidates_three.loc[three]
        # candidates_not_wind = average_net_load.copy()
        # candidates_not_wind = candidates_not_wind.loc[not_wind_anomaly]
        # candidates_not_avg_rel = wind_anomaly.copy()
        # candidates_not_avg_rel = candidates_not_avg_rel.loc[not_average_relative_load]
        # candidates_not_avg_net = wind_anomaly.copy()
        # candidates_not_avg_net = candidates_not_avg_net.loc[not_average_net_load]

        if len(all_three) > 0:
            candidates_three = candidates_three.loc[
                ~np.array([candidates_three.index.overlaps(c) for c in all_three.index]).any(axis=0)
            ]
        # if len(not_wind) > 0:
        #     candidates_not_wind = candidates_not_wind.loc[
        #         ~np.array([candidates_not_wind.index.overlaps(c) for c in not_wind.index]).any(axis=0)
        #     ]
        # if len(not_avg_rel) > 0:
        #     candidates_not_avg_rel = candidates_not_avg_rel.loc[
        #         ~np.array([candidates_not_avg_rel.index.overlaps(c) for c in not_avg_rel.index]).any(axis=0)
        #     ]
        # if len(not_avg_net) > 0:
        #     candidates_not_avg_net = candidates_not_avg_net.loc[
        #         ~np.array([candidates_not_avg_net.index.overlaps(c) for c in not_avg_net.index]).any(axis=0)
        #     ]
        
        # Filter out intervals that overlap with each other.
        # First sort values.
        candidates_three = candidates_three.sort_values("Wind anomaly", ascending=True)
        # candidates_not_wind = candidates_not_wind.sort_values("Net load", ascending=False)
        # candidates_not_avg_rel = candidates_not_avg_rel.sort_values("Wind anomaly", ascending=True)
        # candidates_not_avg_net = candidates_not_avg_net.sort_values("Wind anomaly", ascending=True)

        non_overlapping_I_three = pd.IntervalIndex.from_tuples(
                [], closed="both", dtype="interval[datetime64[ns], both]"
            )
        # non_overlapping_I_not_wind = pd.IntervalIndex.from_tuples(
        #         [], closed="both", dtype="interval[datetime64[ns], both]"
        #     )
        # non_overlapping_I_not_avg_rel = pd.IntervalIndex.from_tuples(
        #         [], closed="both", dtype="interval[datetime64[ns], both]"
        #     )
        # non_overlapping_I_not_avg_net = pd.IntervalIndex.from_tuples(
        #         [], closed="both", dtype="interval[datetime64[ns], both]"
        #     )
        for I in candidates_three.index:
            if not non_overlapping_I_three.overlaps(I).any():
                i = non_overlapping_I_three.searchsorted(I)
                non_overlapping_I_three = non_overlapping_I_three.insert(i, I)
        # for I in candidates_not_wind.index:
        #     if not non_overlapping_I_not_wind.overlaps(I).any():
        #         i = non_overlapping_I_not_wind.searchsorted(I)
        #         non_overlapping_I_not_wind = non_overlapping_I_not_wind.insert(i, I)
        # for I in candidates_not_avg_rel.index:
        #     if not non_overlapping_I_not_avg_rel.overlaps(I).any():
        #         i = non_overlapping_I_not_avg_rel.searchsorted(I)
        #         non_overlapping_I_not_avg_rel = non_overlapping_I_not_avg_rel.insert(i, I)
        # for I in candidates_not_avg_net.index:
        #     if not non_overlapping_I_not_avg_net.overlaps(I).any():
        #         i = non_overlapping_I_not_avg_net.searchsorted(I)
        #         non_overlapping_I_not_avg_net = non_overlapping_I_not_avg_net.insert(i, I)
        for I in non_overlapping_I_three:
            all_three.loc[I] = [wind_anomaly.loc[I, "Wind anomaly"], average_relative_load.loc[I, "Relative load"], average_net_load.loc[I, "Net load"], net_load.loc[I.left:I.right, "Net load"].max(), total_costs_df.loc[I.left:I.right, "0"].sum()]
        # for I in non_overlapping_I_not_wind:
        #     not_wind.loc[I] = [average_relative_load.loc[I, "Relative load"], average_net_load.loc[I, "Net load"]]
        # for I in non_overlapping_I_not_avg_rel:
        #     not_avg_rel.loc[I] = [wind_anomaly.loc[I, "Wind anomaly"], average_net_load.loc[I, "Net load"]]
        # for I in non_overlapping_I_not_avg_net:
        #     not_avg_net.loc[I] = [wind_anomaly.loc[I, "Wind anomaly"], average_relative_load.loc[I, "Relative load"]]

        # Transform all three into the periods format.

    weather_periods = pd.DataFrame()
    for I in all_three.index:
        start = I.left
        end = I.right
        peak_hour = net_load.loc[start:end, "Net load"].idxmax()
        # Only allow for periods outside of July.
        if peak_hour.month < 4 or peak_hour.month > 9:
            weather_periods.loc[I, "start"] = pd.Timestamp(start)
            weather_periods.loc[I, "end"] = pd.Timestamp(end)
            weather_periods.loc[I, "peak_hour"] = pd.Timestamp(peak_hour)
        else:
            continue
    weather_periods.reset_index(inplace=True)
    weather_periods = weather_periods[['start', 'end', 'peak_hour']]

    char_weather_periods = characteristics_sdes(opt_networks, weather_periods)

    stats_weather_periods = stats_sde(
    char_weather_periods,
    char_weather_periods,
    net_load,
    total_load,
    avg_load,
    avg_wind,
    wind_cf,
    wind_caps,
    reindex_opt_objs
    )

    for i, period in weather_periods.iterrows():
        stats_weather_periods.loc[i, "duration"] = (pd.Timestamp(period.end) - pd.Timestamp(period.start)).total_seconds() / 3600
    
    stats_weather_periods = stats_weather_periods[["start", "end","highest_net_load", "avg_net_load", "h2_discharge", "energy_deficit", "max_fc_discharge", "avg_rel_load", "wind_anom", "duration"]]
    
    if overwrite_data:
        stats_weather_periods.to_csv(f"{folder}/weather_filtering/stats_weather_periods.csv")
else:
    stats_weather_periods = pd.read_csv(f"{folder}/weather_filtering/stats_weather_periods.csv", index_col=0)
    stats_weather_periods.start = pd.to_datetime(stats_weather_periods.start)
    stats_weather_periods.end = pd.to_datetime(stats_weather_periods.end)
    
    
 


# %%
import warnings

fig, axs = plt.subplots(1, 8, figsize=(30 * cm, 15 * cm), gridspec_kw={'wspace': 1})

# Suppress FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)

# Reset the index of the heatmap DataFrame
periods_reset = stats_weather_periods.reset_index()

for i, col in enumerate(stats_weather_periods[["highest_net_load", "avg_net_load", "h2_discharge", "duration", "energy_deficit", "max_fc_discharge", "avg_rel_load", "wind_anom"]]):
    sns.stripplot(
        data=periods_reset,
        y=col,
        ax=axs[i],
        jitter=0.1,
        alpha=1,
        size=5,
        palette="tab10",
        legend=False,
    )
    axs[i].set_ylabel(col)

# Add manual legend below plots.


# %% [markdown]
# # Assign predicted clusters from SDEs

# %%
cluster_centroids = pd.read_csv(f"./clustering/{config_name}/centroids_4.csv", index_col=0)
clustered_vals = pd.read_csv(f"./clustering/{config_name}/clustered_vals_4.csv", index_col=0).drop("cluster", axis="columns")
c_cols = cluster_centroids.columns

# %%
# Associate each period to a cluster.
c_wp = stats_weather_periods[c_cols]
normed_c_wp = (c_wp - c_wp.mean()) / c_wp.std()

normalized_vals = (clustered_vals - clustered_vals.mean()) / clustered_vals.std()
n_clusters=4



# %%
kmeans_vals = KMeans(n_clusters=n_clusters, random_state=0).fit(normalized_vals)

# %%
kmeans_vals.predict(normed_c_wp)

# %%
stats_weather_periods["cluster"] = kmeans_vals.predict(normed_c_wp)

# %% [markdown]
# # Candidates: assigned to "most extreme" cluster

# %%
new_candidates_2 = stats_weather_periods[stats_weather_periods.cluster == 2]
unoccurring_years_2 = []
for i, period in new_candidates_2.iterrows():
    start, end = period.start, period.end
    net_y = get_net_year(start)
    shadowed = False
    for j, old_per in stats_periods.iterrows():
        if get_net_year(pd.Timestamp(old_per.start)) == net_y:
            print(f"\n\nThere is another period in {net_y}: {old_per.start} - {old_per.end}")
            display(old_per[["avg_net_load", "highest_net_load",  "h2_discharge", "avg_rel_load", "duration", "wind_anom"]] - period[["avg_net_load", "highest_net_load", "h2_discharge","avg_rel_load", "duration", "wind_anom"]])
            print(f"The total costs in the old period are {total_costs_df.loc[old_per.start:old_per.end, '0'].sum()/1e9:.1f} bn EUR with avg hourly costs of {total_costs_df.loc[old_per.start:old_per.end, '0'].mean()/1e6:.1f} mn EUR")
            print(f"The total costs in the new period are {total_costs_df.loc[start:end, '0'].sum()/1e9:.1f} bn EUR with avg hourly costs of {total_costs_df.loc[start:end, '0'].mean()/1e6:.1f} mn EUR")
            shadowed = True
            old_std = ((all_prices.loc[old_per.start:old_per.end].mean())/all_prices.loc[old_per.start:old_per.end].mean().max()).std()
            new_std = ((all_prices.loc[start:end].mean())/all_prices.loc[start:end].mean().max()).std()
            print(f"The standard deviation of the nodal prices in the old period is {old_std:.4f} and in the new period is {new_std:.4f}")
        else:
            continue
    if not shadowed:
        unoccurring_years_2.append(period)
unocc_2 = pd.DataFrame(unoccurring_years_2)
display(unocc_2[["start", "end", "avg_net_load", "highest_net_load", "energy_deficit", "h2_discharge", "max_fc_discharge", "avg_rel_load", "duration", "wind_anom"]])
for i, period in unocc_2.iterrows():
    start, end = period.start, period.end
    print(f"\n\n The total costs in the new period are {total_costs_df.loc[start:end, '0'].sum()/1e9:.1f} bn EUR with avg hourly costs of {total_costs_df.loc[start:end, '0'].mean()/1e6:.1f} mn EUR")
    new_std = ((all_prices.loc[start:end].mean())/all_prices.loc[start:end].mean().max()).std()
    print(f"The standard deviation of the nodal prices in the period is {new_std:.4f}")

    


# %% [markdown]
# # Candidates: top 10 of KPIs

# %%
criteria = ["avg_net_load", "highest_net_load", "h2_discharge", "max_fc_discharge", "avg_rel_load", "duration"]
new_candidate_periods = []
for crit in criteria:
    new_candidate_periods.extend(stats_weather_periods.sort_values(crit, ascending=False).index[:10])
new_candidate_periods.extend(stats_weather_periods.sort_values("wind_anom", ascending=True).index[:10])
new_candidates = stats_weather_periods.loc[new_candidate_periods].drop_duplicates().sort_index()[["start", "end", "avg_net_load", "highest_net_load", "energy_deficit", "h2_discharge", "max_fc_discharge", "avg_rel_load", "duration", "wind_anom"]]

unoccurring_years = []

for i, period in new_candidates.iterrows():
    start, end = period.start, period.end
    net_y = get_net_year(start)
    shadowed = False
    for j, old_per in stats_periods.iterrows():
        if get_net_year(pd.Timestamp(old_per.start)) == net_y:
            print(f"There is another period in {net_y}: {old_per.start} - {old_per.end}")
            display(old_per[["avg_net_load", "highest_net_load",  "h2_discharge",  "avg_rel_load", "duration", "wind_anom"]] - period[["avg_net_load", "highest_net_load", "h2_discharge",  "avg_rel_load", "duration", "wind_anom"]])
            shadowed = True
            print(f"The total costs in the old period are {total_costs_df.loc[old_per.start:old_per.end, '0'].sum()/1e9:.1f} bn EUR with avg hourly costs of {total_costs_df.loc[old_per.start:old_per.end, '0'].mean()/1e6:.1f} mn EUR")
            print(f"The total costs in the new period are {total_costs_df.loc[start:end, '0'].sum()/1e9:.1f} bn EUR with avg hourly costs of {total_costs_df.loc[start:end, '0'].mean()/1e6:.1f} mn EUR")
            shadowed = True
            old_std = ((all_prices.loc[old_per.start:old_per.end].mean())/all_prices.loc[old_per.start:old_per.end].mean().max()).std()
            new_std = ((all_prices.loc[start:end].mean())/all_prices.loc[start:end].mean().max()).std()
            print(f"The standard deviation of the nodal prices in the old period is {old_std:.4f} and in the new period is {new_std:.4f}")
        else:
            continue
    if not shadowed:
        unoccurring_years.append(period)
unocc = pd.DataFrame(unoccurring_years)
    


# %%
display(unocc[["start", "end", "avg_net_load", "highest_net_load", "energy_deficit", "h2_discharge", "max_fc_discharge", "avg_rel_load", "duration", "wind_anom"]].sort_values("start"))
for i, period in unocc.iterrows():
    start, end = period.start, period.end
    print(f"\n\nThe total costs in the new period are {total_costs_df.loc[start:end, '0'].sum()/1e9:.1f} bn EUR with avg hourly costs of {total_costs_df.loc[start:end, '0'].mean()/1e6:.1f} mn EUR")
    new_std = ((all_prices.loc[start:end].mean())/all_prices.loc[start:end].mean().max()).std()
    print(f"The standard deviation of the nodal prices in the period is {new_std:.4f}")

# %% [markdown]
# # Looking at costs

# %%
for i, period in periods.iterrows():
    start, end = period.start, period.end
    periods.loc[i, "total_cost"] = total_costs_df.loc[start:end, "0"].sum()/1e9
    periods.loc[i, "avg_hourly_cost"] = total_costs_df.loc[start:end, "0"].mean()/1e6


# %%
for i, period in stats_periods.iterrows():
    start, end = period.start, period.end
    stats_periods.loc[i, "total_cost"] = total_costs_df.loc[start:end, "0"].sum()/1e9
    stats_periods.loc[i, "avg_hourly_cost"] = total_costs_df.loc[start:end, "0"].mean()/1e6
    stats_periods.loc[i, "norm_price_std"] = ((all_prices.loc[start:end].mean())/all_prices.loc[start:end].mean().max()).std()

for i, period in new_candidates.iterrows():
    start, end = period.start, period.end
    new_candidates.loc[i, "total_cost"] = total_costs_df.loc[start:end, "0"].sum()/1e9
    new_candidates.loc[i, "avg_hourly_cost"] = total_costs_df.loc[start:end, "0"].mean()/1e6
    new_candidates.loc[i, "norm_price_std"] = ((all_prices.loc[start:end].mean())/all_prices.loc[start:end].mean().max()).std()

for i, period in new_candidates_2.iterrows():
    start, end = period.start, period.end
    new_candidates_2.loc[i, "total_cost"] = total_costs_df.loc[start:end, "0"].sum()/1e9
    new_candidates_2.loc[i, "avg_hourly_cost"] = total_costs_df.loc[start:end, "0"].mean()/1e6
    new_candidates_2.loc[i, "norm_price_std"] = ((all_prices.loc[start:end].mean())/all_prices.loc[start:end].mean().max()).std()


for i, period in stats_weather_periods.iterrows():
    start, end = period.start, period.end
    stats_weather_periods.loc[i, "total_cost"] = total_costs_df.loc[start:end, "0"].sum()/1e9
    stats_weather_periods.loc[i, "avg_hourly_cost"] = total_costs_df.loc[start:end, "0"].mean()/1e6
    stats_weather_periods.loc[i, "norm_price_std"] = ((all_prices.loc[start:end].mean())/all_prices.loc[start:end].mean().max()).std()

# %%
periods.sort_values("avg_hourly_cost")

# %%
stats_periods.describe()

# %%
# Plot heat map for stats_periods with seaborn

# Drop net load peak hour (too similar to highest net load) and wind cf (too similar to wind anomaly)

heatmap = new_candidates

heatmap = heatmap[["highest_net_load", "avg_net_load", "duration","h2_discharge", "max_fc_discharge", "avg_rel_load", "wind_anom","total_cost", "avg_hourly_cost", "norm_price_std"]]

cmaps = [
    "Reds",
    "Reds",
    "Greens",
    #"Greens",
    "Purples",
    "Purples",
    "coolwarm",
    "coolwarm",
    #"Greys",
    "Oranges",
    "Greys",
    "Blues"
]

norms = [
    mcolors.Normalize(vmin=361, vmax=537),
    mcolors.Normalize(vmin=175, vmax=358),
    mcolors.Normalize(vmin=27, vmax=314),
    #mcolors.Normalize(vmin=10, vmax=64),
    mcolors.Normalize(vmin=2.5, vmax=17),
    mcolors.Normalize(vmin=17, vmax=78),
    mcolors.Normalize(vmin=0.84, vmax=1.16),
    mcolors.Normalize(vmin=-0.28, vmax=0.3),
    #mcolors.Normalize(vmin=133, vmax=170),   
    mcolors.Normalize(vmin=100, vmax=130),
    mcolors.Normalize(vmin=317, vmax=3656),
    mcolors.Normalize(vmin=0.05, vmax=0.2),
]

fig, axs = plt.subplots(1, 10, figsize=(30 * cm, 30 * cm))

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



# %%
# Plot heat map for stats_periods with seaborn

# Drop net load peak hour (too similar to highest net load) and wind cf (too similar to wind anomaly)

heatmap = new_candidates_2

heatmap = heatmap[["highest_net_load", "avg_net_load", "duration","h2_discharge", "max_fc_discharge", "avg_rel_load", "wind_anom","total_cost", "avg_hourly_cost", "norm_price_std"]]

cmaps = [
    "Reds",
    "Reds",
    "Greens",
    #"Greens",
    "Purples",
    "Purples",
    "coolwarm",
    "coolwarm",
    #"Greys",
    "Oranges",
    "Greys",
    "Blues"
]

norms = [
    mcolors.Normalize(vmin=361, vmax=537),
    mcolors.Normalize(vmin=175, vmax=358),
    mcolors.Normalize(vmin=27, vmax=314),
    #mcolors.Normalize(vmin=10, vmax=64),
    mcolors.Normalize(vmin=2.5, vmax=17),
    mcolors.Normalize(vmin=17, vmax=78),
    mcolors.Normalize(vmin=0.84, vmax=1.16),
    mcolors.Normalize(vmin=-0.28, vmax=0.3),
    #mcolors.Normalize(vmin=133, vmax=170),   
    mcolors.Normalize(vmin=100, vmax=130),
    mcolors.Normalize(vmin=317, vmax=3656),
    mcolors.Normalize(vmin=0.05, vmax=0.2),
]

fig, axs = plt.subplots(1, 10, figsize=(30 * cm, 7 * cm))

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



# %% [markdown]
# Observations:
# - We can find events during which the three metrics (wind anomaly, average relative load, average net load) are all at the same time more severe than during a previously identified SDE: these are 168 events, however these come from only 9 events: 7, 15, 20, 23, 25, 29, 31, 34. Of these, only 7 and 29 (in cluster 3) have an avg net load above 220 GW. Peak net load is above 450 GW for 7,15,20 and only for those do we use H2 significantly. 7 is a clear wind event, all other ones except 29 (wind anom: -0.11, load: 1.09, also short duration) don't have any interesting fundamentals.
# - In terms of peak net load, average net load, average relative load and also wind anomaly we have identified the most extreme events before already.
# 
# 
# - Most events are relatively long but not very severe; the only relevant events seem to be those that would have been clustered to cluster 2. However most of those come from years where there is a strictly worse SDE, perhaps maybe for 1979/80 and 1946/47 (but these are not that extreme in the first place).
# - Only 1985/86 is interesting as it gives us another event with high hourly costs that was not previosuly identified. It is a wind drought and not really relevant from a load perspective.
# - 2012/13 also gives as another short compound event. The same events we find when looking for Top 10 KPIs.
# 
# - Despite all, looking at average hourly system costs might give us some way of determining severity of an event as these tend to be higher in Cluster 2 which we found consists of the most extreme events. However our filtering by prices and costs does tend to find the most extreme periods; these come from high average net load and/or high peak net load (coming usually from both wind anomaly and high load anomaly). Only looking at the weather does not seem to find more extreme events in any way, and gives us relatively few insights.
# - Usage of fuel cells (in the peak) as well as a high share of the energy deficit covered by fuel cells appear to be good measures of severity of an event. 

# %%

fig, axs = plt.subplots(1, 11, figsize=(30 * cm, 15 * cm), gridspec_kw={'wspace': 1})

# Suppress FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)

# Reset the index of the heatmap DataFrame
periods_reset = stats_weather_periods.reset_index()
p_r = stats_periods.reset_index()

for i, col in enumerate(stats_weather_periods[["highest_net_load", "avg_net_load", "h2_discharge", "duration", "energy_deficit", "max_fc_discharge", "avg_rel_load", "wind_anom","total_cost", "avg_hourly_cost", "norm_price_std"]]):
    sns.stripplot(
        data=periods_reset,
        y=col,
        ax=axs[i],
        jitter=0.1,
        alpha=1,
        size=5,
        color="grey",
        legend=False,
    )
    axs[i].set_ylabel(col)
    sns.stripplot(
        data=p_r,
        y=col,
        ax=axs[i],
        jitter=0.5,
        alpha=0.7,
        size=5,
        color="blue",
        legend=False,
    )
    axs[i].set_ylabel(col)

# Add manual legend below plots.


# %%

fig, axs = plt.subplots(1, 11, figsize=(30 * cm, 15 * cm), gridspec_kw={'wspace': 1})

# Suppress FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)

# Reset the index of the heatmap DataFrame
new_c_reset = new_candidates_2.reset_index()
p_r = stats_periods.reset_index()

for i, col in enumerate(stats_weather_periods[["highest_net_load", "avg_net_load", "h2_discharge", "duration", "energy_deficit", "max_fc_discharge", "avg_rel_load", "wind_anom","total_cost", "avg_hourly_cost", "norm_price_std"]]):
    sns.stripplot(
        data=new_c_reset,
        y=col,
        ax=axs[i],
        jitter=0.1,
        alpha=1,
        size=5,
        color="red",
        legend=False,
    )
    sns.stripplot(
        data=periods_reset,
        y=col,
        ax=axs[i],
        jitter=0.1,
        alpha=0.3,
        size=5,
        color="grey",
        legend=False,
    )
    axs[i].set_ylabel(col)
    sns.stripplot(
        data=p_r,
        y=col,
        ax=axs[i],
        jitter=0.5,
        alpha=0.7,
        size=5,
        color="blue",
        legend=False,
    )
    axs[i].set_ylabel(col)

# Add manual legend below plots.


# %%
for dur in reversed(stats_weather_periods.duration.unique()):
    dur_pers = stats_periods[stats_periods.duration == dur].index

    fig, axs = plt.subplots(1, 11, figsize=(30 * cm, 8 * cm), gridspec_kw={'wspace': 1})
    fig.suptitle(f"Duration: {dur} hours (same as event {list(dur_pers)})")

    # Suppress FutureWarning
    warnings.simplefilter(action='ignore', category=FutureWarning)

    

    # Reset the index of the heatmap DataFrame
    new_c_reset = new_candidates_2.reset_index()
    p_r = stats_periods.reset_index()
    comparable_events = []

    # Add an additional column to list the original SDE in the same year.
    new_c_reset["other_sde"] = 0 
    for i, period in new_c_reset[new_c_reset.duration == dur].iterrows():
        net_year = get_net_year(period.start)
        for j, p in stats_periods.iterrows():
            if get_net_year(pd.Timestamp(p.start)) == net_year:
                new_c_reset.loc[i, "other_sde"] = j
                comparable_events.append(p)
                break
    if len(comparable_events) > 0:
        comparable_events = pd.DataFrame(comparable_events)
        # display(comparable_events)
        # display(new_c_reset[(new_c_reset.duration == dur) & (new_c_reset.other_sde != 0)])

    

    for i, col in enumerate(stats_weather_periods[["highest_net_load", "avg_net_load", "h2_discharge", "duration", "energy_deficit", "max_fc_discharge", "avg_rel_load", "wind_anom","total_cost", "avg_hourly_cost", "norm_price_std"]]):
        sns.stripplot(
            data=new_c_reset[new_c_reset.duration == dur],
            y=col,
            ax=axs[i],
            jitter=0.2,
            alpha=1,
            size=5,
            hue = "other_sde",
            legend=False,
            label = "New candidates (cluster 2)"
        )
        if len(comparable_events) > 0:
            sns.stripplot(
                data = comparable_events,
                y=col,
                ax=axs[i],
                jitter=0.2,
                alpha=1,
                size=5,
                marker="D",
                hue = comparable_events.index,
                legend=False,
                label = "Comparable periods"
            )

        sns.stripplot(
            data=periods_reset[periods_reset.duration == dur],
            y=col,
            ax=axs[i],
            jitter=0.1,
            alpha=0.3,
            size=5,
            color="grey",
            legend=False,
            label = "All `worse` periods",
        )
        axs[i].set_ylabel(col)
        sns.stripplot(
            data=p_r[p_r.duration == dur],
            y=col,
            ax=axs[i],
            jitter=0.5,
            alpha=0.7,
            size=5,
            color="blue",
            marker="X",
            legend=False,
            label = "Original periods"
        )
        axs[i].set_ylabel(col)

    # Add manual legend below plots.
    axs[3].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=3,
    )


# %%
# DUPLICATE?
for dur in reversed(stats_weather_periods.duration.unique()):
    dur_pers = stats_periods[stats_periods.duration == dur].index

    fig, axs = plt.subplots(1, 11, figsize=(30 * cm, 8 * cm), gridspec_kw={'wspace': 1})
    fig.suptitle(f"Duration: {dur} hours (same as event {list(dur_pers)})")

    # Suppress FutureWarning
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # Reset the index of the heatmap DataFrame
    new_c_reset = new_candidates_2.reset_index()
    p_r = stats_periods.reset_index()

    for i, col in enumerate(stats_weather_periods[["highest_net_load", "avg_net_load", "h2_discharge", "duration", "energy_deficit", "max_fc_discharge", "avg_rel_load", "wind_anom","total_cost", "avg_hourly_cost", "norm_price_std"]]):
        sns.stripplot(
            data=new_c_reset[new_c_reset.duration == dur],
            y=col,
            ax=axs[i],
            jitter=0.1,
            alpha=1,
            size=5,
            color="red",
            legend=False,
            label = "New candidates (cluster 2)"
        )
        sns.stripplot(
            data=periods_reset[periods_reset.duration == dur],
            y=col,
            ax=axs[i],
            jitter=0.1,
            alpha=0.3,
            size=5,
            color="grey",
            legend=False,
            label = "All `worse` periods",
        )
        axs[i].set_ylabel(col)
        sns.stripplot(
            data=p_r[p_r.duration == dur],
            y=col,
            ax=axs[i],
            jitter=0.5,
            alpha=0.7,
            size=5,
            color="blue",
            legend=False,
            label = "Original periods"
        )
        axs[i].set_ylabel(col)

    # Add manual legend below plots.
    axs[3].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=3,
    )


# %% [markdown]
# Most events that we have identified seem to be overshadowed by another event in the same year:
# 74 hours (coming from event 29 which is relatively severe, but does not have the highest peak load)
# - 164 (short but not very extreme) is overshadowed by 4 that happened in 46/47 and has a higher peak net load (and is much longer). Average hourly costs are much lower, also because it seems that its geographical extent is smaller (higher std of normed prices).
# - 165 is overshadowed by event 9 (brief but high average net load) in 60/61: slightly higher peak but much lower average net load despite short er duration. Similar wind and load, but much higher average hourly costs and also slightly smaller geographical extent.
# - 166 is overshadowed by 17 in 79/80: slightly lower peak, however higher abg net load but it is also much shorter, so average hourly costs are still lower and so is the geographical extent.
# 
# 94 hours (coming from event 30) which is very severe (wind, net load, peak)
# - 163 is overshadowed bt 30 itself, especially in norm_price std. Still both are severe.
# 
# 235 hours (coming from event 25) which is long, very mild (net load, peak, geog scale, anomalies, load)
# - 150 is overshadowed by 3 in 45/46: lower peak, lower net load, but not std of prices (due to much longer duration)
# - 152 is overshadowed by 11 65/66 which is the most severe event in all points, but 152 is also much longer.
# 
# 260 hours (coming from event 15) which is long, has a relatively high peak but whose avg net load, wind and load anomalies are unremarkable; it is just long and extends far
# - 128 is overshadowed by 21 in 84/85: similar peak (slightly higher), but much lower average net load (also because 21 is shorter), stronger wind anomaly, but by far not as relevant in terms of load anomaly. similar spatial extent. It happened much later during the year, perhaps better solar?
# 
# 
# Not overshadowed were only
# - 161: not very high peak (419), not very high average net load; small spatial extent, low hourly costs
# - 162: peak mid (436), high average net load (272); generally quite expensive 68 bn EUR), large sptial extent and large hourly costs
# both have small FC capacities, and are wind events (162 being much shorter) and very small increase in load.
# 
# 
# 

# %%
stats_periods[["highest_net_load", "avg_net_load", "h2_discharge", "duration", "energy_deficit", "max_fc_discharge", "avg_rel_load", "wind_anom","total_cost", "avg_hourly_cost", "norm_price_std"]].corr()

# %%
stats_periods.sort_values("norm_price_std")

# %%



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

from matplotlib.ticker import (AutoMinorLocator, MultipleLocator, FormatStrFormatter,AutoMinorLocator)
import matplotlib.dates as mdates

from _notebook_utilities import *

import logging

# Suppress warnings and info messages from 'pypsa.io'
logging.getLogger("pypsa.io").setLevel(logging.ERROR)

cm = 1 / 2.54  # centimeters in inches



# %%
regenerate_data = True
overwrite_data = False
config_name = "stressful-weather"

# %%
config, scenario_def, years, opt_networks = load_opt_networks(config_name, load_networks=False)


# %%
periods = load_periods(config)

opt_objs = pd.read_csv(f"processing_data/{config_name}/opt_objs.csv", index_col=0)
reindex_opt_objs = opt_objs.copy().sum(axis="columns") 
reindex_opt_objs.index=years

# Winter load, winter wind cf
net_load = pd.read_csv(f"processing_data/{config_name}/net_load.csv", index_col=0, parse_dates=True)
total_load = pd.read_csv(f"processing_data/{config_name}/total_load.csv", index_col=0, parse_dates=True)
winter_load = pd.read_csv(f"processing_data/{config_name}/winter_load.csv", index_col=0)
winter_cfs = pd.read_csv(f"processing_data/{config_name}/winter_cfs.csv", index_col=0)
annual_cfs = pd.read_csv(f"processing_data/{config_name}/annual_cfs.csv", index_col=0)


# s_caps
s_caps = pd.read_csv(f"processing_data/{config_name}/s_caps.csv", index_col=0)


# %%
folder = f"./load_shedding/{config_name}/design_years"
if regenerate_data:
    mean_system_shedding = {}
    for year in years:
        mean_system_shedding[year] = (pd.read_csv(f"{folder}/{year}/mean_system_shedding.csv", index_col=0, parse_dates=True)/1e3).round(0) # in MW
        mean_system_shedding[year].index = pd.to_datetime(mean_system_shedding[1941].index)
        mean_system_shedding[year].columns = ["load_shedding"]
    mean_system_shedding = pd.concat(mean_system_shedding, axis=1)
    if overwrite_data:
        mean_system_shedding.to_csv(f"load_shedding/{config_name}/design_year_mean_system_shedding.csv")
else:
    mean_system_shedding = pd.read_csv(f"load_shedding/{config_name}/design_year_mean_system_shedding.csv", index_col=0, parse_dates=True)

# %%
folder = f"./load_shedding/{config_name}/design_years"
system_shedding = {}
for year in years:
    system_shedding[year] = (pd.read_csv(f"{folder}/{year}/system_shedding.csv", index_col=0)) # in MW
    # mean_system_shedding[year].index = pd.to_datetime(mean_system_shedding[1941].index)
    system_shedding[year].snapshot = pd.to_datetime(system_shedding[year].snapshot)
    system_shedding[year].columns = ["snapshot","load_shedding"]





# %%
operational_years = {}
for year in years:
    if regenerate_data:      
        op_years = [y for y in years if y != year]
        operational_years[year] = pd.concat([system_shedding[op_year].loc[year] for op_year in op_years], axis=0)
        operational_years[year].set_index("snapshot", inplace=True)
        if overwrite_data:
            operational_years[year].to_csv(f"load_shedding/{config_name}/operational_years/system_shedding_{year}.csv")
    else:
        operational_years[year] = pd.read_csv(f"load_shedding/{config_name}/operational_years/system_shedding_{year}.csv", index_col=0, parse_dates=True)


# %%
# Stats

# Matrix: Design years (rows) x Operational years (columns)
ls_matrix = pd.DataFrame(index = years, columns = years).astype(float)
for year in years:
    for op_year in years:
        if year != op_year:
            ls_matrix.loc[year,op_year] = (system_shedding[year].loc[op_year].load_shedding.sum() / 1e6).round(0) # in GWh
        else:
            ls_matrix.loc[year, op_year] = 0
ls_matrix.columns.name = "Operational year"
ls_matrix.index.name = "Design year"
    

# %%
ls_matrix

# %%
ls_matrix.mean(axis="columns") * 23000 * 1e3 #assuming 23000 EUR/Mwh according to (68) in 6.2.2.2 of https://eepublicdownloads.entsoe.eu/clean-documents/nc-tasks/220225_EB%20Regulation_Art.30_Amendment_ACER%20Decision%20(1).pdf

# %%
((ls_matrix.mean(axis="columns") * 23000 * 1e3)/1e9).plot(kind="bar")

# %%
(reindex_opt_objs + ls_matrix.mean(axis="columns") * 23000 * 1e3).sort_values()

# %%
ls_matrix.T.idxmax().value_counts()

# %%
# Print ls matrix as a heatmap

fig, ax = plt.subplots(figsize=(20 * cm,18 * cm))
sns.heatmap(ls_matrix, annot=False, fmt=".0f", cmap="Blues", ax=ax)
# ax.set_xlabel("Design year");
# ax.set_ylabel("Operational year");
ax.set_title("Load shedding (GWh)");
plt.show()

# %%
# Stats: max load shedding

# Matrix: Design years (rows) x Operational years (columns)
ls_matrix_max = pd.DataFrame(index = years, columns = years).astype(float)
for year in years:
    for op_year in years:
        if year != op_year:
            ls_matrix_max.loc[year,op_year] = (system_shedding[year].loc[op_year].load_shedding.max() / 1e4) # in GW??
        else:
            ls_matrix_max.loc[year, op_year] = 0
ls_matrix_max.columns.name = "Operational year"
ls_matrix_max.index.name = "Design year"
    

# %%
ls_matrix_max.T.idxmax().value_counts()

# %%
ls_matrix_max.T.max()

# %%
# Print ls matrix as a heatmap

fig, ax = plt.subplots(figsize=(20*cm,18*cm))
sns.heatmap(ls_matrix_max, annot=False, fmt=".0f", cmap="Reds", ax=ax)
# ax.set_xlabel("Design year");
# ax.set_ylabel("Operational year");
ax.set_title("Peak load shedding (GW)");
plt.show()

# %%
nodal_flex_p = pd.read_csv(f"processing_data/{config_name}/nodal_flex_p.csv", index_col=[0,1])
system_flex_p = (nodal_flex_p.unstack().sum(axis="rows") / 1e3).round(1)

df_system_flex = system_flex_p.unstack(level=0)
# Reorder df_system_flex to be of the form baseload, and then the order of how we dispatch during extreme events.
df_system_flex = df_system_flex[["battery discharger", "H2 fuel cell"]]

# %%
winter_cfs

# %%
winter_load

# %%
net_load

# %%
# Rank years by difficulty of design year and operational year and compare whether these match.
ranked_years = pd.DataFrame(index = years, columns = ["Design year", "Operational year", "SDE"]).astype(float)
ranked_years["Design year"] = ls_matrix_max.T.describe().mean().sort_values().rank()
ranked_years["Operational year"] = ls_matrix_max.describe().mean().sort_values().rank(ascending=False)
ranked_years["SDE"] = 0
ranked_years["System costs"] = reindex_opt_objs.sort_values().rank(ascending=False)
ranked_years["Battery installed"] = df_system_flex["battery discharger"].rank(ascending=False)
ranked_years["H2 installed"] = df_system_flex["H2 fuel cell"].rank(ascending=False)
ranked_years["Winter load"] = winter_load.sort_values("load").rank(ascending=False)
ranked_years["Winter wind"] = winter_cfs.sort_values("wind").rank(ascending=True)["wind"]

for year in years:
    ranked_years.loc[year, "Highest deficit"] = net_load.loc[f"{year}-07-01":f"{year+1}-06-30 23:00", "Net load"].max()
ranked_years["Highest deficit"] = ranked_years["Highest deficit"].sort_values().rank(ascending=False)

for i, period in periods.iterrows():
    net_year = get_net_year(period.start)
    ranked_years.loc[net_year, "SDE"] = 1

# Scatter plot of ranked years
fig, axs = plt.subplots(1, 3, figsize=(30 * cm, 10 * cm))
fig.suptitle("Average highest load shedding [GW]")
ax = axs[0]
sns.scatterplot(data=ranked_years, x="Design year", y="Operational year", ax=ax, hue="SDE", palette=["lightgrey", "green"])
ax.set_xlabel("Design year with lowest capacity deficit");
ax.set_ylabel("Operational year with highest capacity deficit");
# Add correlation to plot in top left.
ax.text(0.05, 0.95, f"Correlation: {ranked_years[["Design year", "Operational year"]].corr().iloc[0,1]:.2f}", transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax = axs[1]
sns.scatterplot(data=ranked_years, x="Design year", y="System costs", ax=ax, hue="SDE", palette=["lightgrey", "green"])
ax.set_xlabel("Design year with lowest capacity deficit");
ax.set_ylabel("Ranking of system costs");
# Add correlation to plot in top left.
ax.text(0.05, 0.95, f"Correlation: {ranked_years[["Design year", "System costs"]].corr().iloc[0,1]:.2f}", transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax = axs[2]
sns.scatterplot(data=ranked_years, x="Operational year", y="System costs", ax=ax, hue="SDE", palette=["lightgrey", "green"])
ax.set_xlabel("Operational year with highest capacity deficit");
ax.set_ylabel("Ranking of system costs");
# Add correlation to plot in top left.
ax.text(0.05, 0.95, f"Correlation: {ranked_years[["Operational year", "System costs"]].corr().iloc[0,1]:.2f}", transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


# %%
ranked_years = ranked_years.astype(int)
ranked_years.to_csv(f"processing_data/{config_name}/ranked_years.csv")

# %%
ranked_years.sort_values("Design year").T.astype(int)

# %%
ranked_years.sort_values("Operational year").T.astype(int)

# %%
# Plot triangular heatmap
fig, ax = plt.subplots(figsize=(16 * cm,15* cm))

corr = ranked_years.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)


# %%
ranked_years[ranked_years.SDE == 1].mean()

# %%
ranked_years[ranked_years.SDE == 0].mean()

# %%
# Rank years by difficulty of design year and operational year and compare whether these match.
ranked_years_ue = pd.DataFrame(index = years, columns = ["Design year", "Operational year", "SDE"]).astype(float)
ranked_years_ue["Design year"] = ls_matrix.T.describe().mean().sort_values().rank()
ranked_years_ue["Operational year"] = ls_matrix.describe().mean().sort_values().rank(ascending=False)
ranked_years_ue["SDE"] = 0
ranked_years_ue["System costs"] = reindex_opt_objs.sort_values().rank(ascending=False)
ranked_years_ue["Battery installed"] = df_system_flex["battery discharger"].rank(ascending=False)
ranked_years_ue["H2 installed"] = df_system_flex["H2 fuel cell"].rank(ascending=False)
ranked_years_ue["Winter load"] = winter_load.sort_values("load").rank(ascending=False)
ranked_years_ue["Winter wind"] = winter_cfs.sort_values("wind").rank(ascending=True)["wind"]

for year in years:
    ranked_years_ue.loc[year, "Highest deficit"] = net_load.loc[f"{year}-07-01":f"{year+1}-06-30 23:00", "Net load"].max()
ranked_years_ue["Highest deficit"] = ranked_years_ue["Highest deficit"].sort_values().rank(ascending=False)

for i, period in periods.iterrows():
    net_year = get_net_year(period.start)
    ranked_years_ue.loc[net_year, "SDE"] = 1

# Scatter plot of ranked years
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Accumulated load shedding")
ax = axs[0]
sns.scatterplot(data=ranked_years_ue, x="Design year", y="Operational year", ax=ax, hue="SDE", palette=["lightgrey", "green"])
ax.set_xlabel("Design year with least unserved energy");
ax.set_ylabel("Operational year with highest unserved energy");
# Add correlation to plot in top left.
ax.text(0.05, 0.95, f"Correlation: {ranked_years_ue[["Design year", "Operational year"]].corr().iloc[0,1]:.2f}", transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax = axs[1]
sns.scatterplot(data=ranked_years_ue, x="Design year", y="System costs", ax=ax, hue="SDE", palette=["lightgrey", "green"])
ax.set_xlabel("Design year with least unserved energy");
ax.set_ylabel("Ranking of system costs");
# Add correlation to plot in top left.
ax.text(0.05, 0.95, f"Correlation: {ranked_years_ue[["Design year", "System costs"]].corr().iloc[0,1]:.2f}", transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax = axs[2]
sns.scatterplot(data=ranked_years_ue, x="Operational year", y="System costs", ax=ax, hue="SDE", palette=["lightgrey", "green"])
ax.set_xlabel("Operational year with highest unserved energy");
ax.set_ylabel("Ranking of system costs");
# Add correlation to plot in top left.
ax.text(0.05, 0.95, f"Correlation: {ranked_years_ue[["Operational year", "System costs"]].corr().iloc[0,1]:.2f}", transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# %%
ranked_years_ue.sort_values("Design year").T.astype(int)

# %%
ranked_years_ue.sort_values("Operational year").T.astype(int)

# %%
ranked_years_ue.sort_values("System costs").T.astype(int)

# %%
ranked_years_ue[ranked_years_ue.SDE == 1].mean()

# %%
ranked_years_ue[ranked_years_ue.SDE == 0].mean()

# %%
ranked_years.columns

# %%
annual_cfs.sort_values("solar").rank(ascending=False)["solar"]

# %%
classification_years = ranked_years.copy()
classification_years.rename(columns={"Design year": "Prevents peaks", "Operational year": "Causes peaks", "Highest deficit": "Highest net load"}, inplace=True)
classification_years["Causes deficit"] = ranked_years_ue["Operational year"].astype(int)
classification_years["Prevents deficits"] = ranked_years_ue["Design year"].astype(int)
classification_years["Lowest solar CF"] = annual_cfs.sort_values("solar").rank(ascending=True)["solar"].astype(int)
classification_years["Lowest wind CF"] = annual_cfs.sort_values("wind").rank(ascending=True)["wind"].astype(int)

# Reorder according to the following categories:
# - Annual values: total system costs, solar cf, wind cf
# - Winter / weather: winter load, winter wind
# - Operational: causes deficit, prevents deficits
# - SDE: SDE, highest net load, prevents peaks, causes peaks
# - System installation: battery, H2
classification_years = classification_years[["System costs", "Lowest solar CF", "Lowest wind CF", "Winter load", 
#"Winter wind", 
"Causes deficit", "Prevents deficits", "SDE", "Highest net load", "Prevents peaks", "Causes peaks", #"Battery installed", "H2 installed"
]]

# %%
# Only keep years that have a value below 6 (except SDE)
filtered_classification_years = classification_years.drop("SDE", axis="columns")
filtered_classification_years = filtered_classification_years[filtered_classification_years < 5].dropna(how="all")

# %%
filtered_classification_years["SDE"] = classification_years["SDE"]

# %%
filtered_classification_years = filtered_classification_years.T

# %%
# Plot seabornheatmap of classification years with breaks for each category.

fig, axs = plt.subplots(2,1, figsize=(22 * cm, 10* cm), sharex=True, gridspec_kw={"height_ratios": [12,1]})
fig.suptitle("Classification of years")

norm = mpl.colors.Normalize(vmin=1, vmax=5)
norm_bin = mpl.colors.Normalize(vmin=0.1, vmax=1, clip=True)
# Take first three colours of tab20c color map.
cmap = "Blues_r"

ax = axs[0]
sns.heatmap(filtered_classification_years.loc[["System costs", "Lowest solar CF", "Lowest wind CF", "Winter load", 
#"Winter wind", 
"Causes deficit", "Prevents deficits", "Highest net load", "Prevents peaks", "Causes peaks", #"Battery installed", "H2 installed"
]], 
cmap=cmap, ax=ax, cbar=False, annot=True)


ax = axs[1]
sns.heatmap(filtered_classification_years.loc[["SDE"]], cmap="Greens", ax=ax, cbar=False)



for ax in axs:
    ax.set_xlabel("")
    # Set length of y tick markers to 0.
    ax.tick_params(axis="both", length=0)
    # Use all indices for y ticks.
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha="right")
    #ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")




# %%
# Plot triangular heatmap
fig, ax = plt.subplots(figsize=(16 * cm,15* cm))

corr = ranked_years_ue.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)


# %%
# Worst periods
all_years_index = operational_years[1941].index
# Remove Feb 28
all_years_index = all_years_index.append(pd.date_range("1941-07-01", "1942-06-30 23:00", freq="h"))
all_years_index = all_years_index.sort_values()

# %%
worst_periods = pd.DataFrame(index = all_years_index, columns = years).astype(float)
worst_periods.columns.name = "Design year"

for year in years:
    df_helper = system_shedding[year].copy()
    helper_index = operational_years[year].index
    df_helper = df_helper.set_index("snapshot")
    df_helper.index = helper_index   
    worst_periods[year] = df_helper.load_shedding
worst_periods = worst_periods.fillna(0).round(0)

# %%
worst_periods = worst_periods.round(0)

# %%
worst_hours_per_netw = pd.DataFrame(worst_periods.idxmax().value_counts())
worst_hours_per_netw["period"] = None

for hour in worst_hours_per_netw.index:
    for i, period in periods.iterrows():
        if hour in pd.date_range(period.start, period.end, freq="h"):
            worst_hours_per_netw.loc[hour, "period"] = period.name
            
display(worst_hours_per_netw)

# %%
# e.g. the event in 1942 seems mostly load driven (very cold winter) and not as much by a wind deficit. It wasn't identified because the winter per se was very difficult (shorter fuel cell spikes, and low installed fuel cell capacities). In that year the difficulty was mostly addressed through high wind capacities, and not as much through high flexibility investments.

# %%
worst_periods.max()

# %%



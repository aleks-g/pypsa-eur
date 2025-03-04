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

def in_or_on_hull(p, hull, threshold=0):
    from matplotlib.path import Path
    hull_path = Path(hull.points[hull.vertices])
    return hull_path.contains_point(p, radius=threshold)

def draw_region(
    shape,
    cluster_nr,
    T,
    print_progress=False,
):
    # Define first convex hull to include all points in the selected cluster
    subnodes = shape[shape.cluster == cluster_nr]
    centroid = subnodes[["x", "y"]].mean()
    hull = ConvexHull(subnodes[["x", "y"]])

    # Find all subregions which lie within the convex hull.
    subregions = []
    for i in shape.index:
        if in_or_on_hull(shape.loc[i][["x", "y"]].values, hull):
            subregions.append(i)
    subregions = shape.loc[subregions]

    # Compute first threshold.
    threshold = len(subregions[subregions.cluster == cluster_nr])/len(subregions)
    iters = 0

    # Iteratively reduce the convex hull to meet threshold.
    while threshold < T and iters < 20:
        if print_progress:
            print(f"Reduce cluster, as threshold {T} is not met; current ratio: {threshold} for {len(subregions)} included nodes.")
        helper_df = subregions[subregions.cluster == cluster_nr].copy()
        # Sort all nodes inside the chosen cluster by distance from the centroid.
        helper_df["distance"] = helper_df.apply(lambda x: geopy.distance.distance((x.y, x.x), (centroid.y, centroid.x)).m, axis=1)
        helper_df = helper_df.sort_values("distance", ascending=False)
        # Remove the node furthest away from the centroid.
        helper_df.drop(helper_df.index[0], inplace=True)
        # Recompute the centroid.
        centroid = helper_df[["x", "y"]].mean()
        # Recompute the convex hull.
        hull = ConvexHull(helper_df[["x", "y"]])

        # Find all subregions which lie within the convex hull.
        subregions = []
        for i in shape.index:
            if in_or_on_hull(shape.loc[i][["x", "y"]].values, hull):
                subregions.append(i)
        subregions = shape.loc[subregions]
        threshold = len(subregions[subregions.cluster == cluster_nr])/len(subregions)
        iters += 1
        if iters == 20:
            if print_progress:
                print("Max iterations reached.")
            break
    if print_progress:
        print(f"Threshold: {threshold} with {len(subregions)} included clusters.")
    # Check if it's possible to add any neighbouring nodes within our cluster without violating the threshold.

    helper_df = subregions.copy()
    outside_cluster = shape[shape.cluster == cluster_nr].index.difference(subregions.index)
    for i in outside_cluster:
        helper_df.loc[i] = shape.loc[i]
        temp_hull = ConvexHull(helper_df[["x", "y"]])
        subr_tmp = []
        for j in shape.index:
            if in_or_on_hull(shape.loc[j][["x", "y"]].values, temp_hull):
                subr_tmp.append(j)
        subr_tmp = shape.loc[subr_tmp]
        if len(subr_tmp[subr_tmp.cluster == cluster_nr])/len(subr_tmp) > threshold:
            subregions = subr_tmp
            threshold = len(subregions[subregions.cluster == cluster_nr])/len(subregions)
            hull = temp_hull
    if print_progress:
        print(f"Final threshold: {threshold} with {len(subregions)} included clusters.")
    return subregions, hull

def select_clusters(
    r,
    column,
    cluster_nb,
    cluster_sense,
):
    # Cluster the data. 
    kmeans = KMeans(n_clusters=cluster_nb, random_state=0).fit(r[column].values.reshape(-1,1))
    r["cluster"] = kmeans.labels_
    centres = kmeans.cluster_centers_

    if cluster_sense == "max":
        cluster = centres.argmax()
    elif cluster_sense == "min":
        cluster = centres.argmin()
    else:
        raise ValueError("Cluster sense must be either 'max' or 'min'. Others have not been implemented.")
    return r, cluster



def plot_clustered_wind(
    config_name,
    period,
    event_nr,
    df,
    cluster_sense,
    tech,
    norm,
    regions,
    offshore_regions,
    projection,
    n,
    cluster_nb = 3,
    threshold = 0.9,
    averages = None,
    ax = None,
    save = False,
    cmap = "coolwarm",
):
    """Note: this uses anomalies"""
    if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw={"projection": projection})
    n.plot(ax=ax, bus_sizes=0, line_widths=0.5, link_widths=0.5)

    start = period.start
    end = period.end

    # Shift to access averages.
    shifted_start = f"1942{str(start)[4:]}" if start.month <7 else f"1941{str(start)[4:]}"
    shifted_end = f"1942{str(end)[4:]}" if end.month <7 else f"1941{str(end)[4:]}"

    # Set up GeoDataFrame.
    r = regions.set_index("name")
    r["x"], r["y"] = n.buses.x, n.buses.y
    r = gpd.geodataframe.GeoDataFrame(r, crs="EPSG:4326")
    r = r.to_crs(projection.proj4_init)

    if averages is None:
        raise ValueError("Averages must be provided if anomalies are to be computed.")
    anomalies = df.loc[start:end].mean() - averages.loc[shifted_start:shifted_end].mean()

    # Need to separate onshore and offshore.
    onshore_anomalies = anomalies.filter(like="onwind", axis=0)
    offshore_anomalies = anomalies.filter(like="offwind", axis=0)
    onshore_anomalies.index = n.generators.loc[onshore_anomalies.index].bus.values
    offshore_anomalies.index = n.generators.loc[offshore_anomalies.index].bus.values
    # Merge ac, dc, float.
    offshore_anomalies = offshore_anomalies.groupby(offshore_anomalies.index).mean()

    # Add additional dataframe for offshore
    r_off = offshore_regions.set_index("name")
    r_off["x"], r_off["y"] = n.buses.x, n.buses.y
    r_off = gpd.geodataframe.GeoDataFrame(r_off, crs="EPSG:4326")
    r_off = r_off.to_crs(projection.proj4_init)

    # Separate on- and offshore wind.
    r["wind"] = onshore_anomalies
    r_off["offwind"] = offshore_anomalies

    r_off.plot(ax=ax,
        column="offwind",
        cmap=cmap,
        norm=norm,
        alpha=0.6,
        linewidth=0,
        zorder=1,
        )
    r.plot(ax=ax,
            column=tech,
            cmap=cmap,
            norm=norm,
            alpha=0.6,
            linewidth=0,
            zorder=1,
            )
    r, cluster = select_clusters(r, tech, cluster_nb, cluster_sense)
    sns.scatterplot(
            x="x",
            y="y",
            data=r,
            hue="cluster",
            palette="tab10",
            s=100,
            ax=ax,
            zorder=2,
            legend=False,
        )
    subregions, hull = draw_region(r, cluster, threshold, print_progress=False)
    if save:
        vertices = pd.DataFrame(hull.points[hull.vertices], columns=["x", "y"])
        if use_anomalies:
            vertices.to_csv(f"processing_data/{config_name}/maps/{tech}_anom/hull_{threshold}_event{event_nr}.csv")
        else:
            vertices.to_csv(f"processing_data/{config_name}/maps/{tech}/hull_{threshold}_event{event_nr}.csv")

    # Plot the convex hull.
    for simplex in hull.simplices:
        ax.plot(hull.points[simplex, 0], hull.points[simplex, 1], "k-")
    return ax

def plot_clustered_map(
    config_name,
    period,
    event_nr,
    df,
    cluster_sense,
    tech,
    norm,
    regions,
    projection,
    n,
    cluster_nb = 3,
    threshold = 0.9,
    use_anomalies = False,
    averages = None,
    ax = None,
    save = False,
    cmap = "coolwarm",
):
    if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw={"projection": projection})
    n.plot(ax=ax, bus_sizes=0, line_widths=0.5, link_widths=0.5)

    start = period.start
    end = period.end

    # Shift to access averages.
    shifted_start = f"1942{str(start)[4:]}" if start.month <7 else f"1941{str(start)[4:]}"
    shifted_end = f"1942{str(end)[4:]}" if end.month <7 else f"1941{str(end)[4:]}"

    # Set up GeoDataFrame.
    r = regions.set_index("name")
    r["x"], r["y"] = n.buses.x, n.buses.y
    r = gpd.geodataframe.GeoDataFrame(r, crs="EPSG:4326")
    r = r.to_crs(projection.proj4_init)

    if use_anomalies:
        if averages is None:
            raise ValueError("Averages must be provided if anomalies are to be computed.")
        anomalies = df.loc[start:end].mean() - averages.loc[shifted_start:shifted_end].mean()
        if tech == "solar":
            anomalies.index = n.generators.loc[anomalies.index].bus.values
            anomalies = anomalies.groupby(anomalies.index).mean()
            r[tech] = anomalies
        elif tech == "load":
            anomalies /= averages.loc[shifted_start:shifted_end].mean()
            r[tech] = anomalies
        else:
            r[tech] = anomalies
    else:
        r[tech] = df.loc[start:end].mean()

    r.plot(ax=ax,
        column=tech,
        cmap=cmap,
        norm=norm,
        alpha=0.6,
        linewidth=0,
        zorder=1,
        )
    r, cluster = select_clusters(r, tech, cluster_nb, cluster_sense)
    sns.scatterplot(
            x="x",
            y="y",
            data=r,
            hue="cluster",
            palette="tab10",
            s=100,
            ax=ax,
            zorder=2,
            legend=False,
        )
    subregions, hull = draw_region(r, cluster, threshold, print_progress=False)
    if save:
        vertices = pd.DataFrame(hull.points[hull.vertices], columns=["x", "y"])
        if use_anomalies:
            vertices.to_csv(f"processing_data/{config_name}/maps/{tech}_anom/hull_{threshold}_event{event_nr}.csv")
        else:
            vertices.to_csv(f"processing_data/{config_name}/maps/{tech}/hull_{threshold}_event{event_nr}.csv")

    # Plot the convex hull.
    for simplex in hull.simplices:
        ax.plot(hull.points[simplex, 0], hull.points[simplex, 1], "k-")
    return ax


def plot_affected_areas(
    config_name,
    period,
    event_nr,
    hulls,
    techs,
    pretty_names,
    colours,
    fill_df,
    fill_tech,
    fill_norm,
    fill_cmap,
    regions,
    n,
    projection,
    ax = None,
    save=False,
):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw={"projection": projection})
    n.plot(ax=ax, bus_sizes=0, bus_colors="black", line_widths=0, link_widths=0, link_colors="black", line_colors="black",color_geomap=True)

    start = period.start
    end = period.end

    r = regions.set_index("name")
    r["x"], r["y"] = n.buses.x, n.buses.y
    r = gpd.geodataframe.GeoDataFrame(r, crs="EPSG:4326")
    r = r.to_crs(projection.proj4_init)

    r[fill_tech] = fill_df.loc[start:end].mean()

    r.plot(ax=ax,
        column=fill_tech,
        cmap=fill_cmap,
        norm=fill_norm,
        alpha=0.6,
        linewidth=0,
        zorder=1,
        )
    
    # Add cbar.
    sm = plt.cm.ScalarMappable(cmap=fill_cmap, norm=fill_norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation="vertical", pad=0.01, aspect=20)
    cbar.set_label("Fuel cell usage", fontsize=7)
    ticks = [0, 0.25, 0.5, 0.75, 1]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t:.0%}" for t in ticks], fontsize=6)
    
    legend_elements = []
    hatches = [None, None, None, "x", "/", "o", ".", "*", "O", "-"]

    for hull, tech, colour, pretty_name, hatch in zip(hulls, techs, colours, pretty_names, hatches):
        # For now, only edges, no filling.
        patch = Polygon(xy = hull.points[hull.vertices], closed=True, ec = colour, fill = False, lw = 1, zorder=2)
        legend_elements.append(Patch(ec = colour, fill = False, lw = 1, label=pretty_name))
        ax.add_patch(patch)
    return legend_elements

def grid_wind(
    config_name,
    periods,
    df,
    cluster_sense,
    tech,
    norm,
    regions,
    offshore_regions,
    projection,
    n,
    cluster_nb = 3,
    threshold = 0.9,
    averages = None,
    save = False,
    cmap = "coolwarm",
):
    """Note: this uses anomalies"""
    nb_rows = len(periods) // 3 if len(periods) % 3 == 0 else len(periods) // 3 + 1

    fig, axs = plt.subplots(nb_rows, 3, figsize=(18, 6 * nb_rows), subplot_kw={"projection": projection})

    for i, period in enumerate(periods.iterrows()):
        ax = axs[i // 3, i % 3]
        
        plot_clustered_wind(
            config_name,
            periods.loc[i],
            i,
            df,
            cluster_sense,
            tech,
            norm,
            regions = regions,
            offshore_regions = offshore_regions,
            projection = projection,
            n = n,
            cluster_nb = cluster_nb,
            threshold = threshold,
            averages = averages,
            ax = ax,
            save = save,
            cmap = cmap,
        )
        ax.set_title(f"Event {i}: {periods.loc[i, "start"]} - {periods.loc[i, "end"]}")
    plt.tight_layout()
    plt.show()
    plt.close()

    if save:
        fig.savefig(f"processing_data/{config_name}/maps/{tech}_anom/clustered_{tech}_anom_{cluster_sense}.pdf", bbox_inches="tight")


def grid_maps(
    config_name,
    periods,
    df,
    cluster_sense,
    tech,
    norm,
    regions,
    projection,
    n,
    cluster_nb = 3,
    threshold = 0.9,
    use_anomalies = False,
    averages = None,
    save = False,
    cmap = "coolwarm",
):
    nb_rows = len(periods) // 3 if len(periods) % 3 == 0 else len(periods) // 3 + 1
    fig, axs = plt.subplots(nb_rows, 3, figsize=(18, 6 * nb_rows), subplot_kw={"projection": projection})
    for i, period in enumerate(periods.iterrows()):
        ax = axs[i // 3, i % 3]
        
        plot_clustered_map(
            config_name,
            periods.loc[i],
            i,
            df,
            cluster_sense,
            tech,
            norm,
            regions = regions,
            projection = projection,
            n = n,
            cluster_nb = cluster_nb,
            threshold = threshold,
            use_anomalies = use_anomalies,
            averages = averages,
            ax = ax,
            save = save,
            cmap = cmap,
        )
        ax.set_title(f"{periods.loc[i, "start"]} - {periods.loc[i, "end"]}")
    plt.tight_layout()
    plt.show()
    plt.close()

    if save:
        if use_anomalies:
            fig.savefig(f"processing_data/{config_name}/maps/{tech}_anom/clustered_{tech}_anom_{cluster_sense}.pdf", bbox_inches="tight")
        else:
            fig.savefig(f"processing_data/{config_name}/maps/{tech}/clustered_{tech}_{cluster_sense}.pdf", bbox_inches="tight")

def grid_affected_areas(
    config_name,
    periods,
    hulls_collection,
    techs,
    pretty_names,
    colours,
    fill_df,
    fill_tech,
    fill_norm,
    fill_cmap,
    regions,
    n,
    projection,
    save=False,
):
    nb_rows = len(periods) // 3 if len(periods) % 3 == 0 else len(periods) // 3 + 1
    fig, axs = plt.subplots(nb_rows, 3, figsize=(18, 6 * nb_rows), subplot_kw={"projection": projection})

    for (i, period) in enumerate(periods.iterrows()):
        ax = axs[i // 3, i % 3]
        hulls_period = [hulls_collection[tech][i] for tech in techs]
        
        legend_elements = plot_affected_areas(
            config_name,
            period[1],
            i,
            hulls_period,
            techs,
            pretty_names,
            colours,
            fill_df,
            fill_tech,
            fill_norm,
            fill_cmap,
            regions,
            n,
            projection,
            ax = ax,
            save = save,
        )
        ax.set_title(f"Event {i}: {periods.loc[i, "start"]} - {periods.loc[i, "end"]}")
    
    # Add legend to the last row.
    ax_l = axs[-1, 0]
    ax_l.legend(handles = legend_elements, ncols = 4, bbox_to_anchor=(2, -0.1), loc='upper center')

    if save:
        fig.savefig(f"processing_data/{config_name}/system_maps/affected_areas.pdf", bbox_inches="tight")


# def plot_cluster(
#         period,
#         event_nr,
#         df,
#         cluster_sense,
#         tech,
#         norm,
#         regions,
#         projection,
#         n,
#         cluster_nb = 3,
#         threshold = 0.9,
#         use_anomalies = False,
#         averages = None,
#         ax = None,
#         save = False,
#         cmap = "coolwarm",
#     ):
#         if ax is None:
#             fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw={"projection": projection})
#         n.plot(ax=ax, bus_sizes=0, line_widths=0.5, link_widths=0.5)

#         start = period.start
#         end = period.end


#         # Shift to access averages.
#         shifted_start = f"1942{str(start)[4:]}" if start.month <7 else f"1941{str(start)[4:]}"
#         shifted_end = f"1942{str(end)[4:]}" if end.month <7 else f"1941{str(end)[4:]}"

#         # Set up GeoDataFrame.
#         r = regions.set_index("name")
#         r["x"], r["y"] = n.buses.x, n.buses.y
#         r = gpd.geodataframe.GeoDataFrame(r, crs="EPSG:4326")
#         r = r.to_crs(projection.proj4_init)


#         if use_anomalies:
#             if averages is None:
#                 raise ValueError("Averages must be provided if anomalies are to be computed.")
#             anomalies = df.loc[start:end].mean() - averages.loc[shifted_start:shifted_end].mean()
#             if tech == "wind":
#                 # Need to separate onshore and offshore.
#                 onshore_anomalies = anomalies.filter(like="onwind", axis=0)
#                 offshore_anomalies = anomalies.filter(like="offwind", axis=0)
#                 onshore_anomalies.index = n.generators.loc[onshore_anomalies.index].bus.values
#                 offshore_anomalies.index = n.generators.loc[offshore_anomalies.index].bus.values
#                 # Merge ac, dc, float.
#                 offshore_anomalies = offshore_anomalies.groupby(offshore_anomalies.index).mean()

#                 # Add additional dataframe for offshore
#                 r_off = offshore_regions.set_index("name")
#                 r_off["x"], r_off["y"] = n.buses.x, n.buses.y
#                 r_off = gpd.geodataframe.GeoDataFrame(r_off, crs="EPSG:4326")
#                 r_off = r_off.to_crs(projection.proj4_init)

#                 # Separate on- and offshore wind.
#                 r["wind"] = onshore_anomalies
#                 r_off["offwind"] = offshore_anomalies

#                 r_off.plot(ax=ax,
#                     column="offwind",
#                     cmap=cmap,
#                     norm=norm,
#                     alpha=0.6,
#                     linewidth=0,
#                     zorder=1,
#                     )
#             elif tech == "solar":
#                 anomalies.index = n.generators.loc[anomalies.index].bus.values
#                 anomalies = anomalies.groupby(anomalies.index).mean()
#                 r[tech] = anomalies
#             elif tech == "load":
#                 anomalies /= averages.loc[shifted_start:shifted_end].mean()
#                 r[tech] = anomalies
#             elif tech == "storage":
#                 anomalies.index = n.storage_units.loc[anomalies.index].bus.values
#                 anomalies = anomalies.groupby(anomalies.index).sum()
#                 storage_caps = n.storage_units.p_nom_opt * n.storage_units.max_hours
#                 storage_caps = storage_caps.groupby(n.storage_units.bus.values).sum()
#                 # Only keep storage units with a capacity of at least 1 GWh.
#                 storage_caps = storage_caps[storage_caps > 1e3]
#                 anomalies /= storage_caps
#                 anomalies.fillna(0, inplace=True)
#                 r[tech] = anomalies
#                 r[tech].fillna(0, inplace=True)
#             else:
#                 r[tech] = anomalies
#         else:
#             r[tech] = df.loc[start:end].mean()

#         r.plot(ax=ax,
#             column=tech,
#             cmap=cmap,
#             norm=norm,
#             alpha=0.6,
#             linewidth=0,
#             zorder=1,
#             )
#         r, cluster = select_clusters(r, tech, cluster_nb, cluster_sense)

#         sns.scatterplot(
#             x="x",
#             y="y",
#             data=r,
#             hue="cluster",
#             palette="tab10",
#             s=100,
#             ax=ax,
#             zorder=2,
#             legend=False,
#         )

#         subregions, hull = draw_region(r, cluster, threshold, print_progress=False)
#         if save:
#             vertices = pd.DataFrame(hull.points[hull.vertices], columns=["x", "y"])
#             if use_anomalies:
#                 vertices.to_csv(f"processing_data/{config_name}/maps/{tech}_anom/hull_{threshold}_event{event_nr}.csv")
#             else:
#                 vertices.to_csv(f"processing_data/{config_name}/maps/{tech}/hull_{threshold}_event{event_nr}.csv")

#         # Plot the convex hull.
#         for simplex in hull.simplices:
#             ax.plot(hull.points[simplex, 0], hull.points[simplex, 1], "k-")
#         return ax

# def plot_all_clusters(
#     periods,
#     df,
#     cluster_sense,
#     tech,
#     norm,
#     regions,
#     projection,
#     n,
#     cluster_nb = 3,
#     threshold = 0.9,
#     use_anomalies = False,
#     averages = None,
#     save = False,
#     cmap = "coolwarm",
# ):
#     nb_rows = len(periods) // 3 if len(periods) % 3 == 0 else len(periods) // 3 + 1

#     fig, axs = plt.subplots(nb_rows, 3, figsize=(18, 6 * nb_rows), subplot_kw={"projection": projection})

#     for i, period in enumerate(periods.iterrows()):
#         ax = axs[i // 3, i % 3]
        
#         plot_cluster(
#             periods.loc[i],
#             i,
#             df,
#             cluster_sense,
#             tech,
#             norm,
#             regions = regions,
#             projection = projection,
#             n = n,
#             cluster_nb = cluster_nb,
#             threshold = threshold,
#             use_anomalies = use_anomalies,
#             averages = averages,
#             ax = ax,
#             save = save,
#             cmap = cmap,
#         )
#         ax.set_title(f"{periods.loc[i, "start"]} - {periods.loc[i, "end"]}")
#     plt.tight_layout()
#     plt.show()
#     plt.close()

#     if save:
#         if use_anomalies:
#             fig.savefig(f"processing_data/{config_name}/maps/{tech}_anom/clustered_{tech}_anom_{cluster_sense}.pdf", bbox_inches="tight")
#         else:
#             fig.savefig(f"processing_data/{config_name}/maps/{tech}/clustered_{tech}_{cluster_sense}.pdf", bbox_inches="tight")


# def nodal_weighted_flex(nodal_flex_periods, nodal_flex_p, start, end):
#     net_year = get_net_year(start)
#     new_df = pd.DataFrame()
#     # Hard-coded fix for negative values in AC.
#     nodal_flex_periods["AC"] = nodal_flex_periods["AC"].abs().clip(upper=1)
#     for i in nodal_flex_periods.index.levels[0]:
#         new_df[i] = (nodal_flex_periods.loc[i].loc[start:end] * nodal_flex_p.loc[i].loc[net_year]).mean()
#     return new_df.T

# def nodal_annual_caps(nodal_flex_p, start, end):
#     net_year = get_net_year(start)
#     new_df = pd.DataFrame()
#     for i in nodal_flex_p.index.levels[0]:
#         new_df[i] = nodal_flex_p.loc[i].loc[net_year]
#     return new_df.T.round(0)

# def plot_flex_cluster(
#         period,
#         event_nr,
#         df,
#         capacities,
#         cluster_sense,
#         tech,
#         norm,
#         regions,
#         projection,
#         n,
#         cluster_nb = 3,
#         threshold = 0.9,
#         ax = None,
#         save = False,
#         cmap = "coolwarm",
#     ):
#         if ax is None:
#             fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw={"projection": projection})
#         n.plot(ax=ax, bus_sizes=0, line_widths=0.5, link_widths=0.5)

#         start = period.start
#         end = period.end

#         # Shift to access averages.
#         shifted_start = f"1942{str(start)[4:]}" if start.month <7 else f"1941{str(start)[4:]}"
#         shifted_end = f"1942{str(end)[4:]}" if end.month <7 else f"1941{str(end)[4:]}"

#         # Set up GeoDataFrame.
#         r = regions.set_index("name")
#         r["x"], r["y"] = n.buses.x, n.buses.y
#         r = gpd.geodataframe.GeoDataFrame(r, crs="EPSG:4326")
#         r = r.to_crs(projection.proj4_init)
#         if tech == "transmission":
#             flex_usage = nodal_weighted_flex(df, capacities, start, end)[["DC", "AC"]].sum(axis=1)/nodal_annual_caps(capacities, start, end)[["DC", "AC"]].sum(axis=1)
#         elif tech == "dispatch":
#             flex_usage = nodal_weighted_flex(df, capacities, start, end)[["biomass", "nuclear"]].sum(axis=1)/nodal_annual_caps(capacities, start, end)[["biomass", "nuclear"]].sum(axis=1)
#             flex_usage.fillna(0, inplace=True)
#         elif tech == "discharge":
#             flex_usage = nodal_weighted_flex(df, capacities, start, end)[["PHS", "hydro"]].sum(axis=1)/nodal_annual_caps(capacities, start, end)[["PHS", "hydro"]].sum(axis=1)
#             flex_usage.fillna(0, inplace=True)
        
#         r[tech] = flex_usage

#         r.plot(ax=ax,
#             column=tech,
#             cmap=cmap,
#             norm=norm,
#             alpha=0.6,
#             linewidth=0,
#             zorder=1,
#             )
#         r, cluster = select_clusters(r, tech, cluster_nb, cluster_sense)

#         sns.scatterplot(
#             x="x",
#             y="y",
#             data=r,
#             hue="cluster",
#             palette="tab10",
#             s=100,
#             ax=ax,
#             zorder=2,
#             legend=False,
#         )

#         subregions, hull = draw_region(r, cluster, threshold, print_progress=False)
#         if save:
#             vertices = pd.DataFrame(hull.points[hull.vertices], columns=["x", "y"])
#             vertices.to_csv(f"processing_data/{config_name}/maps/{tech}/hull_{threshold}_event{event_nr}.csv")

#         # Plot the convex hull.
#         for simplex in hull.simplices:
#             ax.plot(hull.points[simplex, 0], hull.points[simplex, 1], "k-")
#         return ax

# def plot_flex_all_clusters(
#     periods,
#     df,
#     capacities,
#     cluster_sense,
#     tech,
#     norm,
#     regions,
#     projection,
#     n,
#     cluster_nb = 3,
#     threshold = 0.9,
#     save = False,
#     cmap = "coolwarm",
# ):
#     nb_rows = len(periods) // 3 if len(periods) % 3 == 0 else len(periods) // 3 + 1

#     fig, axs = plt.subplots(nb_rows, 3, figsize=(18, 6 * nb_rows), subplot_kw={"projection": projection})

#     for i, period in enumerate(periods.iterrows()):
#         ax = axs[i // 3, i % 3]
        
#         plot_flex_cluster(
#             periods.loc[i],
#             i,
#             df,
#             capacities,
#             cluster_sense,
#             tech,
#             norm,
#             regions = regions,
#             projection = projection,
#             n = n,
#             cluster_nb = cluster_nb,
#             threshold = threshold,
#             ax = ax,
#             save = save,
#             cmap = cmap,
#         )
#         ax.set_title(f"{periods.loc[i, "start"]} - {periods.loc[i, "end"]}")
#     plt.tight_layout()
#     plt.show()
#     plt.close()

#     if save:
#         fig.savefig(f"processing_data/{config_name}/maps/{tech}/clustered_{tech}_{cluster_sense}.pdf", bbox_inches="tight")


# def affected_areas(
#     periods,
#     hulls,
#     techs,
#     pretty_names,
#     colours,
#     n,
#     projection,
#     save=False,
# ):
#     nb_rows = len(periods) // 3 if len(periods) % 3 == 0 else len(periods) // 3 + 1

#     fig, axs = plt.subplots(nb_rows, 3, figsize=(18, 6 * nb_rows), subplot_kw={"projection": projection})

#     assert len(hulls) == len(techs)

#     hatches = [None, None, None, "x", "/", "o", ".", "*", "O", "-"]

#     for i, period in enumerate(periods.iterrows()):
#         ax = axs[i // 3, i % 3]

#         n.plot(ax=ax, bus_sizes=0.005, bus_colors="black", line_widths=0.05, link_widths=0.05, link_colors="black", line_colors="black",color_geomap=True)

#         legend_elements = []

#         for hull, tech, colour, pretty_name, hatch in zip(hulls, techs, colours, pretty_names, hatches):
#             if tech == "wind_anom" or tech == "load_anom":
#                 patch = Polygon(xy = hull[i].points[hull[i].vertices], closed=True, fc=colour, alpha=0.5, zorder = 1)
#                 legend_elements.append(Patch(facecolor=colour, alpha=0.5, label=pretty_name))
#             else:
#                 patch = Polygon(xy = hull[i].points[hull[i].vertices], closed=True, ec = colour, fill = False, lw = 3, zorder=2, hatch=hatch)
#                 legend_elements.append(Patch(ec = colour, fill = False, lw = 3, label=pretty_name, hatch=hatch))
#             ax.add_patch(patch)
#             ax.set_title(f"{periods.loc[i, "start"]} - {periods.loc[i, "end"]}")

#     # Add legend to the last row.
#     ax_l = axs[-1, 0]
#     ax_l.legend(handles = legend_elements, ncols = 4, bbox_to_anchor=(2, -0.1), loc='upper center') 

#     if save:
#         fig.savefig(f"processing_data/{config_name}/system_maps/affected_areas.pdf", bbox_inches="tight")


# def plot_region_fill(
#     tech,
#     start,
#     end,
#     n,
#     ax
# ):
#     if tech == "fuel_cells":
#         norm = fc_norm
#         cmap = "Purples"
#         regions = onshore_regions
#         df = fc_flex
    
    
#     r = regions.set_index("name")
#     r["x"], r["y"] = n.buses.x, n.buses.y
#     r = gpd.geodataframe.GeoDataFrame(r, crs="EPSG:4326")
#     r = r.to_crs(projection.proj4_init)
#     r[tech] = df.loc[start:end].mean()

#     r.plot(ax=ax,
#         column = tech,
#         cmap = cmap,
#         norm = norm,
#         alpha=0.6,
#         linewidth=0,
#         zorder=1)
#     return ax


# def affected_areas2(
#     periods,
#     hulls,
#     techs,
#     pretty_names,
#     colours,
#     reg_fill,
#     n,
#     projection,
#     save=False,
# ):
#     nb_rows = len(periods) // 3 if len(periods) % 3 == 0 else len(periods) // 3 + 1

#     fig, axs = plt.subplots(nb_rows, 3, figsize=(18, 6 * nb_rows), subplot_kw={"projection": projection})

#     assert len(hulls) == len(techs)

#     hatches = [None, None, None, "x", "/", "o", ".", "*", "O", "-"]

#     for (i, period) in enumerate(periods.iterrows()):
#         ax = axs[i // 3, i % 3]

#         n.plot(ax=ax, bus_sizes=0, bus_colors="black", line_widths=0, link_widths=0, link_colors="black", line_colors="black",color_geomap=True)

#         # Fill usage of fuel cells.
#         plot_region_fill(reg_fill, period[1].start, period[1].end, n, ax)

#         legend_elements = []

#         for hull, tech, colour, pretty_name, hatch in zip(hulls, techs, colours, pretty_names, hatches):
#             # if tech == "wind_anom" or tech == "load_anom":
#             #     patch = Polygon(xy = hull[i].points[hull[i].vertices], closed=True, fc=colour, alpha=0.5, zorder = 1)
#             #     legend_elements.append(Patch(facecolor=colour, alpha=0.5, label=pretty_name))
#             # else:
#             patch = Polygon(xy = hull[i].points[hull[i].vertices], closed=True, ec = colour, fill = False, lw = 3, zorder=2, hatch=hatch)
#             legend_elements.append(Patch(ec = colour, fill = False, lw = 3, label=pretty_name, hatch=hatch))
#             ax.add_patch(patch)
#             ax.set_title(f"{periods.loc[i, "start"]} - {periods.loc[i, "end"]}")

#     # Add legend to the last row.
#     ax_l = axs[-1, 0]
#     ax_l.legend(handles = legend_elements, ncols = 4, bbox_to_anchor=(2, -0.1), loc='upper center') 

#     if save:
#         fig.savefig(f"processing_data/{config_name}/system_maps/affected_areas.pdf", bbox_inches="tight")



    



if __name__ == "__main__":
    config_name = "stressful-weather"
    # folder = f"./processing_data/{config_name}"

    # config, scenario_def, years, opt_networks = load_opt_networks(config_name, load_networks=False)

    # periods = load_periods(config)
    # prices = pd.read_csv(f"{folder}/marginal_prices.csv", index_col=0, parse_dates=True)

    

    # # Load onshore and offshore regions for shapefile.
    # onshore_regions = gpd.read_file(f"../resources/{config_name}/weather_year_1941/regions_onshore_base_s_90.geojson")
    # offshore_regions = gpd.read_file(f"../resources/{config_name}/weather_year_1941/regions_offshore_base_s_90.geojson")

    # # Load one network for reference and the layout.
    # n = pypsa.Network("../results/{config_name}/weather_year_1941/networks/base_s_90_elec_lc1.25_Co2L.nc")


    # # Load the means, anomalies.
    # load_means = pd.read_csv("../results/means/load_1941-2020_100bn_12-336h_90_elec_lc1.25_Co2L.csv", index_col=0, parse_dates=True)
    # solar_means = pd.read_csv("../results/means/solar_1941-2020_100bn_12-336h_90_elec_lc1.25_Co2L.csv", index_col=0, parse_dates=True)
    # wind_means = pd.read_csv("../results/means/wind_1941-2020_100bn_12-336h_90_elec_lc1.25_Co2L.csv", index_col=0, parse_dates=True)

    # # Load costs
    # total_costs_df = pd.read_csv(f"{folder}/total_costs.csv", index_col=[0,1])
    # total_costs = {}
    # for year in years:
    #     df = total_costs_df.loc[year]
    #     df.index = pd.to_datetime(df.index)


    #     # Load flexibility indicators.
    #     # Usage of flexibility indicators.
    #     nodal_flex_u = xr.open_dataset(f"processing_data/nodal_flex_u.nc")
    #     # Extracted for periods.
    #     nodal_flex_periods = pd.read_csv(f"{folder}/nodal_flex_periods.csv", index_col=[0,1], parse_dates=True)
    #     # Seasonality of nodal flexibility.
    #     nodal_flex_seasonality = pd.read_csv(f"processing_data/nodal_flex_seasonality.csv", index_col=[0,1])

    #     # Capacities.
    #     nodal_flex_p = pd.read_csv(f"{folder}/nodal_flex_p.csv", index_col=[0,1])

    #     # Anomalies during periods and peak hours in flexibility usage.
    #     nodal_flex_anomaly_periods = pd.read_csv(f"{folder}/nodal_flex_anomaly_periods.csv", index_col=[0,1], parse_dates=True)
    #     nodal_flex_anomaly_peak = pd.read_csv(f"{folder}/nodal_flex_anomaly_peak_hour.csv", index_col=[0,1], parse_dates=True)

    #     # Capacity factors and capacities.
    #     wind_cf = xr.open_dataset(f"{folder}/wind_cf.nc").to_dataframe()
    #     solar_cf = xr.open_dataset(f"{folder}/solar_cf.nc").to_dataframe()
    #     wind_caps = pd.read_csv(f"{folder}/wind_caps.csv", index_col=0)
    #     solar_caps = pd.read_csv(f"{folder}/solar_caps.csv", index_col=0)

    #     # Load laod.
    #     total_load = pd.read_csv(f"{folder}/total_load.csv", index_col=0, parse_dates=True)

    #     # Load state of charge
    #     avg_soc = pd.read_csv(f"{folder}/avg_soc.csv", index_col=0, parse_dates=True)
    #     state_of_charge = pd.read_csv(f"{folder}/state_of_charge.csv", index_col=0, parse_dates=True)

    
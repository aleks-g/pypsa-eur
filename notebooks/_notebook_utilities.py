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

import seaborn


from matplotlib.ticker import (AutoMinorLocator, MultipleLocator, FormatStrFormatter,AutoMinorLocator)
import matplotlib.dates as mdates
from matplotlib.path import Path
from matplotlib.lines import Line2D

from typing import NamedTuple, Optional



## COMPUTATIONS AND INITIATION

cm = 1/2.54


def load_opt_networks(
    config_name: str,
    config_str: str = "base_s_90_elec_lc1.25_Co2L",
    load_networks: bool = True,
):
    '''Load the configuration, scenario definition, years and optimal networks.'''

    # Load the configuration
    with open(f"../config/{config_name}.yaml", "r") as file:
        config = yaml.safe_load(file)
    
    # Load scenario definition
    scenario_file = config["run"]["scenarios"]["file"]
    with open(f"../{scenario_file}", "r") as f:
        scenario_def = yaml.safe_load(f)

    # Load years
    scenario_names = list(scenario_def.keys())
    scenarios = {int(sn.split("_")[-1]): sn for sn in scenario_names}
    years = list(scenarios.keys())

    # Load optimal networks
    if load_networks:
        opt_networks = {
        year: pypsa.Network(f"../results/{config_name}/{scenario_name}/networks/{config_str}.nc")
        for year, scenario_name in scenarios.items()
        }
    else:
        opt_networks = None

    return config, scenario_def, years, opt_networks


def load_periods(config: dict):
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

def compute_all_duals(
    opt_networks: dict,
    storage_units: bool = False,
    by_node: bool = False,
):
    '''Compute the duals for the electricity costs, storage costs and fuel cell costs.'''
    years = list(opt_networks.keys())


    # Electricity costs
    all_prices = pd.concat([
        opt_networks[y].buses_t["marginal_price"] for y in years
    ])
    nodal_costs = {
        y: opt_networks[y].buses_t["marginal_price"] * opt_networks[y].loads_t["p_set"]
        for y in years
    }

    # Storage costs
    if storage_units:
        nodal_storage_costs = {
            y: opt_networks[y].storage_units_t["mu_energy_balance"] * opt_networks[y].storage_units_t["state_of_charge"]
        for y in years
        }
    else:
        nodal_storage_costs = {
            y: opt_networks[y].stores_t["mu_energy_balance"] * opt_networks[y].stores_t["e"]
        for y in years
        }
    
    # Fuel cell costs, only if storage units are not present

    if not storage_units:
        prod = {}
        shadow_price_fc = {}
        for y, n in opt_networks.items():
            fc_i = n.links.filter(like="H2 Fuel Cell", axis="rows").index
            prod[y] = n.links_t.p0.loc[:,fc_i]
            # Set to 0, whenever production is below 0.1 [MW].
            prod[y] = prod[y].where(prod[y] > 0.1, 0)
            shadow_price_fc[y] = -n.links_t.mu_upper.loc[:,fc_i]
        
        nodal_fc_costs = {y: prod[y] * shadow_price_fc[y] for y in years}




    if by_node:
        if storage_units:
            return nodal_costs, nodal_storage_costs
        else:
            return nodal_costs, nodal_storage_costs, nodal_fc_costs
    

    else:
        total_costs = {y: C.sum(axis="columns") for y, C in nodal_costs.items()}
        total_storage_costs = {y: C.sum(axis="columns") for y, C in nodal_storage_costs.items()}
        if storage_units:
            return total_costs, total_storage_costs
        else:
            total_fc_costs = {y: C.sum(axis="columns") for y, C in nodal_fc_costs.items()}
            return total_costs, total_storage_costs, total_fc_costs

def annual_variable_cost(
    n: pypsa.Network,
) -> float:
    """Compute the annual variable costs in a PyPSA network `n`. Don't count load shedding."""
    weighting = n.snapshot_weightings.objective
    total = 0
    # Add variable costs for generators
    i = n.generators.loc[n.generators.carrier != "load-shedding"].index
    total += (
        n.generators_t.p[i].multiply(weighting, axis=0).sum(axis=0)
        * n.generators.marginal_cost
    ).sum()
    # Add variable costs for links (lines have none), in our model all 0 though?
    total += (
        n.links_t.p0[n.links.index].abs().multiply(weighting, axis=0).sum(axis=0)
        * n.links.marginal_cost
    ).sum()
    # Add variable costs for stores
    total += (
        n.stores_t.p[n.stores.index].abs().multiply(weighting, axis=0).sum(axis=0)
        * n.stores.marginal_cost
    ).sum()
    # Add variable costs for storage units
    total += (
        n.storage_units_t.p[n.storage_units.index]
        .abs()
        .multiply(weighting, axis=0)
        .sum(axis=0)
        * n.storage_units.marginal_cost
    ).sum()
    # Divide by the number of years the network is defined over. We disregard
    # leap years.
    total /= n.snapshot_weightings.objective.sum() / 8760
    return total


def optimal_costs(
    opt_networks: dict,
    techs: list = ["variable", "H2", "battery", "transmission-ac", "transmission-dc", "onwind", "offwind", "solar"],
    pretty_names: dict = {"variable": "Variable costs", "transmission-ac": "AC  transmission", "transmission-dc": "DC transmission",
              "onwind": "Onshore wind", "offwind": "Offshore wind", "solar": "Solar", "H2": "Hydrogen", "battery": "Battery"},
    storage_units: bool = False,
):
    '''Compute the optimal costs for the optimal networks. Some hard-coded feature'''
    years = list(opt_networks.keys())
    opt_objs = pd.DataFrame(index=years, columns=techs)
    vres = ["onwind", "offwind-ac", "offwind-dc", "solar", "offwind-float", "solar-hsat"]

    for y, n in opt_networks.items():
        # Generators
        g_inv = n.generators["p_nom_opt"] * n.generators["capital_cost"]
        for g in vres:
            i = n.generators.loc[n.generators.carrier == g].index
            opt_objs.loc[y, g] = g_inv.loc[i].sum()
        # Transmission:
        opt_objs.loc[y, "transmission-ac"] = ((n.lines.s_nom_opt - n.lines.s_nom) * n.lines.capital_cost).sum()
        opt_objs.loc[y, "transmission-dc"] = ((n.links.loc[n.links.carrier.str.contains("DC")].p_nom_opt - n.links.p_nom) * n.links.loc[n.links.carrier.str.contains("DC")].capital_cost).sum()
        # Storage
        if storage_units:
            s_inv = n.storage_units["p_nom_opt"] * n.storage_units["capital_cost"]
            for s in ["H2", "battery"]:
                i = n.storage_units.loc[n.storage_units.carrier == s].index
                opt_objs.loc[y, s] = s_inv.loc[i].sum()
        else:
            # Stores instead of storage units
            s_inv = n.stores["e_nom_opt"] * n.stores["capital_cost"]
            for s in ["H2", "battery"]:
                i = n.stores.loc[n.stores.carrier == s].index
                opt_objs.loc[y, s] = s_inv.loc[i].sum()
            # Add charging and discharging capacities.
            s_c_inv = n.links["p_nom_opt"] * n.links["capital_cost"]
            for s in ["H2", "battery"]:
                i = n.links.loc[n.links.carrier.str.contains(s)].index
                opt_objs.loc[y, s] += s_c_inv.loc[i].sum()
        # Variable:
        opt_objs.loc[y, "variable"] = annual_variable_cost(n)
    
    opt_objs["offwind"] = opt_objs["offwind-ac"] + opt_objs["offwind-dc"] + opt_objs["offwind-float"]
    opt_objs["solar"] = opt_objs["solar"] + opt_objs["solar-hsat"]
    del opt_objs["offwind-ac"]
    del opt_objs["offwind-dc"]
    del opt_objs["offwind-float"]
    del opt_objs["solar-hsat"]

    # Also compile the total objective values as a sanity check; n.objective
    # should be equal to the sum of the individual objective values.
    obj_totals = pd.DataFrame(index=years, columns=["total"])
    obj_totals["total"] = [n.objective for n in opt_networks.values()]

    opt_objs = opt_objs.rename(pretty_names, axis='columns')

    # Change the index to the form "80/81"
    opt_objs.index = [f"{str(y)[-2:]}/{str(y+1)[-2:]}" for y in opt_objs.index]

    return opt_objs, obj_totals


# Get network for a given date. Here, opt_networks[n] is defined over the period n-07-01 to (n+1)-06-30.
def get_net_year(date):
    year = date.year
    if date.month < 7:
        year -= 1
    return year

def get_year_period(row):
    return get_net_year(row["start"])
    


## FLEXIBILITY CALCULATIONS

# Flexibility in use (system-wide)
def calculate_transmission_in_use(n):
    # Transmission links
    dc_i = n.links[n.links.carrier == "DC"].index
    dc_p_nom = n.links.loc[dc_i].p_nom_opt
    dc_in_use = ((n.links_t.p0.loc[:, dc_i].abs())/(0.99 * dc_p_nom)).clip(upper=1)

    # Transmission lines
    ac_i = n.lines.index
    ac_in_use = ((n.lines_t.p0.loc[:, ac_i].abs())/(0.99 * n.lines_t.p0.loc[:,ac_i].max())).clip(upper=1)

    # Combine DC and AC transmission data
    transmission_in_use = pd.concat([dc_in_use, ac_in_use], axis=1)
    transmission_p_nom = pd.concat([dc_p_nom, n.lines_t.p0.loc[:,ac_i].max()], axis=0)
    
    return transmission_in_use, transmission_p_nom

def calculate_dispatchable_in_use(n):
    # Dispatchable technologies
    biomass_i = n.generators[n.generators.carrier == "biomass"].index
    nuclear_i = n.generators[n.generators.carrier == "nuclear"].index
    ror_i = n.generators[n.generators.carrier == "ror"].index

    biomass_p_nom = n.generators.loc[biomass_i].p_nom_opt
    nuclear_p_nom = n.generators.loc[nuclear_i].p_nom_opt
    ror_p_nom = n.generators.loc[ror_i].p_nom_opt
    dispatchable_p_nom = pd.concat([biomass_p_nom, nuclear_p_nom, ror_p_nom], axis=0)

    # Calculate in-use ratios for dispatchable technologies
    biomass_in_use = (n.generators_t.p.loc[:,biomass_i] / (0.99 * biomass_p_nom)).clip(upper=1)
    nuclear_in_use = (n.generators_t.p.loc[:,nuclear_i] / (0.99 * nuclear_p_nom)).clip(upper=1)
    ror_in_use = (n.generators_t.p.loc[:,ror_i] / ror_p_nom * n.generators_t.p_max_pu.loc[:,ror_i])

    dispatchable_in_use = pd.concat([biomass_in_use, nuclear_in_use, ror_in_use], axis=1)
    
    return dispatchable_in_use, dispatchable_p_nom

def calculate_storage_discharge_in_use(n):
    # Storage discharge
    fuel_cells_i = n.links[n.links.carrier == "H2 fuel cell"].index
    battery_i = n.links[n.links.carrier == "battery discharger"].index
    phs_i = n.storage_units[n.storage_units.carrier == "PHS"].index
    hydro_i = n.storage_units[n.storage_units.carrier == "hydro"].index

    fuel_cells_p_nom = n.links.loc[fuel_cells_i].p_nom_opt * n.links.loc[fuel_cells_i].efficiency 
    battery_p_nom = n.links.loc[battery_i].p_nom_opt * n.links.loc[battery_i].efficiency
    phs_p_nom = n.storage_units.loc[phs_i].p_nom_opt
    hydro_p_nom = n.storage_units.loc[hydro_i].p_nom_opt
    storage_p_nom = pd.concat([fuel_cells_p_nom, battery_p_nom, phs_p_nom, hydro_p_nom], axis=0)

    # Calculate in-use ratios for storage discharge
    fuel_cells_in_use = (n.links_t.p1.loc[:,fuel_cells_i].abs() / (0.99 * fuel_cells_p_nom)).clip(upper=1)
    battery_in_use = (n.links_t.p1.loc[:,battery_i].abs() / (0.99 * battery_p_nom)).clip(upper=1)
    phs_in_use = (n.storage_units_t.p.loc[:,phs_i] / (0.99 * phs_p_nom)).clip(upper=1)
    hydro_in_use = (n.storage_units_t.p.loc[:,hydro_i] / (0.99 * hydro_p_nom)).clip(upper=1)

    storage_discharge_in_use = pd.concat([fuel_cells_in_use, battery_in_use, phs_in_use, hydro_in_use], axis=1)
    
    return storage_discharge_in_use, storage_p_nom


def coarse_system_flex(m):
    transmission_in_use, transmission_p_nom = calculate_transmission_in_use(m)
    dispatchable_in_use, dispatchable_p_nom = calculate_dispatchable_in_use(m)
    storage_discharge_in_use, storage_p_nom = calculate_storage_discharge_in_use(m)

    system_flex_coarse = pd.DataFrame(columns = ["transmission", "dispatchable", "storage_discharge"], index = transmission_in_use.index)

    system_flex_coarse["transmission"] = (transmission_in_use * transmission_p_nom).sum(axis=1)/transmission_p_nom.sum()
    system_flex_coarse["dispatchable"] = (dispatchable_in_use * dispatchable_p_nom).sum(axis=1)/dispatchable_p_nom.sum()
    system_flex_coarse["storage_discharge"] = (storage_discharge_in_use * storage_p_nom).sum(axis=1)/storage_p_nom.sum()
    
    return system_flex_coarse

def detailed_system_flex(m):
    transmission_in_use, transmission_p_nom = calculate_transmission_in_use(m)
    dispatchable_in_use, dispatchable_p_nom = calculate_dispatchable_in_use(m)
    storage_discharge_in_use, storage_p_nom = calculate_storage_discharge_in_use(m)

    system_flex_detailed = pd.DataFrame(columns = ["DC", "AC", "biomass", "nuclear", "ror", "fuel_cells", "battery", "phs", "hydro"], index = transmission_in_use.index)

    dc_i = m.links[m.links.carrier == "DC"].index
    ac_i = m.lines.index
    biomass_i = m.generators[m.generators.carrier == "biomass"].index
    nuclear_i = m.generators[m.generators.carrier == "nuclear"].index
    ror_i = m.generators[m.generators.carrier == "ror"].index
    fuel_cells_i = m.links[m.links.carrier == "H2 fuel cell"].index
    battery_i = m.links[m.links.carrier == "battery discharger"].index
    phs_i = m.storage_units[m.storage_units.carrier == "PHS"].index
    hydro_i = m.storage_units[m.storage_units.carrier == "hydro"].index

    biomass_p_nom = m.generators.loc[biomass_i].p_nom_opt
    nuclear_p_nom = m.generators.loc[nuclear_i].p_nom_opt
    ror_p_nom = m.generators.loc[ror_i].p_nom_opt
    fuel_cells_p_nom = m.links.loc[fuel_cells_i].p_nom_opt
    battery_p_nom = m.links.loc[battery_i].p_nom_opt
    phs_p_nom = m.storage_units.loc[phs_i].p_nom_opt
    hydro_p_nom = m.storage_units.loc[hydro_i].p_nom_opt

    system_flex_detailed["DC"] = (transmission_in_use[dc_i] * transmission_p_nom[dc_i]).sum(axis=1)/transmission_p_nom[dc_i].sum()
    system_flex_detailed["AC"] = (transmission_in_use[ac_i] * transmission_p_nom[ac_i]).sum(axis=1)/transmission_p_nom[ac_i].sum()
    system_flex_detailed["biomass"] = (dispatchable_in_use[biomass_i] * biomass_p_nom).sum(axis=1)/biomass_p_nom.sum()
    system_flex_detailed["nuclear"] = (dispatchable_in_use[nuclear_i] * nuclear_p_nom).sum(axis=1)/nuclear_p_nom.sum()
    system_flex_detailed["ror"] = (dispatchable_in_use[ror_i] * ror_p_nom).sum(axis=1)/ror_p_nom.sum()
    system_flex_detailed["fuel_cells"] = (storage_discharge_in_use[fuel_cells_i] * fuel_cells_p_nom).sum(axis=1)/fuel_cells_p_nom.sum()
    system_flex_detailed["battery"] = (storage_discharge_in_use[battery_i] * battery_p_nom).sum(axis=1)/battery_p_nom.sum()
    system_flex_detailed["phs"] = (storage_discharge_in_use[phs_i] * phs_p_nom).sum(axis=1)/phs_p_nom.sum()
    system_flex_detailed["hydro"] = (storage_discharge_in_use[hydro_i] * hydro_p_nom).sum(axis=1)/hydro_p_nom.sum()

    return system_flex_detailed


def nodal_flexibility(
    opt_networks: dict,
    nodes: list,
    tech: list = ["DC", "AC", "biomass", "nuclear", "ror", "H2 fuel cell", "battery discharger", "PHS", "hydro"],
    ):

    nodal_flex_u = {}
    nodal_flex_p = {}

    transmission_in_use = {}
    transmission_p_nom = {}
    dispatchable_in_use = {}
    dispatchable_p_nom = {}
    storage_discharge_in_use = {}
    storage_p_nom = {}

    for y, n in opt_networks.items():
        transmission_in_use[y], transmission_p_nom[y] = calculate_transmission_in_use(n)
        dispatchable_in_use[y], dispatchable_p_nom[y] = calculate_dispatchable_in_use(n)
        storage_discharge_in_use[y], storage_p_nom[y] = calculate_storage_discharge_in_use(n)

    for node in nodes:
        flex_p = pd.DataFrame(columns=tech)
        list_flex_u = []
        for y, n in opt_networks.items():
            flex_u = pd.DataFrame(columns=tech, index = n.snapshots)
            for t in tech:
                # Transmission
                if t == "DC":
                    i = n.links[(n.links.carrier == "DC") & ((n.links.bus0 == node) | (n.links.bus1 == node))].index
                    flex_p.loc[y, t] = n.links.loc[i, "p_nom_opt"].sum()
                    flex_u.loc[:, t] = ((transmission_in_use[y].loc[:,i] * n.links.loc[i, "p_nom_opt"]).sum(axis="columns")) / flex_p.loc[y,t]
                elif t == "AC":
                    i = n.lines[((n.lines.bus0 == node) | (n.lines.bus1 == node))].index
                    flex_p.loc[y, t] = n.lines.loc[i, "s_nom_opt"].sum()
                    flex_u.loc[:, t] = ((transmission_in_use[y].loc[:,i] * n.lines.loc[i, "s_nom_opt"]).sum(axis="columns")) / flex_p.loc[y,t]
                # Dispatch
                elif t in ["biomass", "nuclear", "ror"]:
                    i = n.generators[(n.generators.bus == node) & (n.generators.carrier == t)].index
                    flex_p.loc[y, t] = n.generators.loc[i, "p_nom_opt"].sum()
                    flex_u.loc[:, t] = ((dispatchable_in_use[y].loc[:,i] * n.generators.loc[i, "p_nom_opt"]).sum(axis="columns")) / flex_p.loc[y,t]
                # Storage
                else:
                    if t in ["H2 fuel cell", "battery discharger"]:
                        i = n.links[(n.links.carrier == t) & (n.links.bus1 == node)].index
                        flex_p.loc[y, t] = (n.links.loc[i, "p_nom_opt"] * n.links.loc[i, "efficiency"]).sum()
                        flex_u.loc[:, t] = ((storage_discharge_in_use[y].loc[:,i] * n.links.loc[i, "p_nom_opt"] * n.links.loc[i, "efficiency"]).sum(axis="columns")) / flex_p.loc[y,t]
                    elif t in ["PHS", "hydro"]:
                        i = n.storage_units[(n.storage_units.carrier == t) & (n.storage_units.bus == node)].index
                        flex_p.loc[y, t] = n.storage_units.loc[i, "p_nom_opt"].sum()
                        flex_u.loc[:, t] = ((storage_discharge_in_use[y].loc[:,i] * n.storage_units.loc[i, "p_nom_opt"]).sum(axis="columns")) / flex_p.loc[y,t]
                    else:
                        print(f"{t} not implemented.")
                        continue
                flex_u = flex_u.astype(float).round(2)
            list_flex_u.append(flex_u)
        nodal_flex_p[node] = flex_p
        nodal_flex_u[node] = pd.concat(list_flex_u)
    return nodal_flex_p, nodal_flex_u

# NOTE: For ror, we only have limited availability in the different time steps, so the flexibility availability in capacities can be slightly misleading.





## PLOTTING
def plot_optimal_costs(
    opt_networks: dict,
    techs: list = ["variable", "H2", "battery", "transmission-ac", "transmission-dc", "onwind", "offwind", "solar"],
    pretty_names: dict = {"variable": "Variable costs", "transmission-ac": "AC  transmission", "transmission-dc": "DC transmission",
              "onwind": "Onshore wind", "offwind": "Offshore wind", "solar": "Solar", "H2": "Hydrogen", "battery": "Battery"},
    storage_units: bool = False,
    save_fig: bool = False,
):
    opt_objs, _ = optimal_costs(opt_networks,techs,pretty_names,storage_units)
    n_cs = list(opt_networks.values())[0].carriers.color
    cs = ["#c0c0c0", n_cs["H2"], n_cs["battery"], "#70af1d", "#92d123",  n_cs["onwind"], n_cs["offwind-ac"], n_cs["solar"]]

    fig, ax = plt.subplots(1,1, figsize=(32.0*cm, 7*cm))
    (opt_objs / 1e9).plot.bar(stacked=True, ax=ax, color=cs, width=0.7)

    # Labels
    ax.set_xlabel("Weather year")
    ax.set_ylabel("Annual system costs [billion EUR / a]")

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(reversed(handles), reversed(labels), bbox_to_anchor=(0, -0.25), loc='upper left', ncol=3, fontsize=9, frameon=False)

    # Ticks, grid
    ax.yaxis.set_major_locator(MultipleLocator(50))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.yaxis.set_minor_locator(MultipleLocator(10))
    ax.yaxis.grid(color='lightgray', linestyle='solid', which='major')
    ax.yaxis.grid(color='lightgray', linestyle='dotted', which='minor')

    if save_fig:
        plt.savefig(f"../plots/optimal_costs.pdf", bbox_inches='tight')
    else:
        plt.show()

# Plot a segmented horizontal line (using LineCollection) where each segment is coloured according to total_costs[y]
def plot_segmented_line(ax, x, y, c, cmap, norm, **kwargs):
    segments = np.array([x[:-1], y[:-1], x[1:], y[1:]]).T.reshape(-1, 2, 2)
    # Log norm
    lc = mpl.collections.LineCollection(segments, cmap=cmap, norm=norm, **kwargs)
    lc.set_array(c)
    ax.add_collection(lc)
    return lc

def plot_duals(
        periods: pd.DataFrame, 
        years: list, 
        left_vals: dict, 
        right_vals: dict, 
        left_norm,
        right_norm, 
        left_cmap: str = "Blues",
        right_cmap: str = "Oranges",
        left_str: str = "Fuel cell costs",
        right_str: str = "Electricity costs",
        left_ticks: list = [1, 10, 100],
        right_ticks: list = [1, 10, 100, 1000, 10000],
        left_scaling: float = 1e-6,
        right_scaling: float = 1e-6,
        save_fig: bool = False,
        path_str: str = None,
        alt_periods: Optional[pd.DataFrame] = None,
    ):
    fig, axs = plt.subplots(1, 2, figsize=(18 * cm, 18 * cm))

    dates = mdates.date2num(pd.date_range("2014-10-01", "2015-03-31", freq="D").to_pydatetime())

    for ax, year_selection in zip(axs, [years[:(len(years)//2)], years[(len(years)//2):]]):

        for y in year_selection:
            # Resample to daily resolution.
            C = left_vals[y].resample("D").mean()
            # Drop leap days from C
            C = C.loc[(C.index.month != 2) | (C.index.day != 29)]
            # Select only the period from October to March inclusive.
            C = C.loc[(C.index.month >= 10) | (C.index.month <= 3)]
            plot_segmented_line(
                ax, dates, [y - 0.25] * len(C), C * left_scaling, left_cmap, left_norm, lw=4
            )

            # Resample to daily resolution.
            alt_C = right_vals[y].resample("D").mean()
            # Drop leap days from C
            alt_C = alt_C.loc[(alt_C.index.month != 2) | (alt_C.index.day != 29)]
            # Select only the period from October to March inclusive.
            alt_C = alt_C.loc[(alt_C.index.month >= 10) | (alt_C.index.month <= 3)]
            plot_segmented_line(
                ax, dates, [y + 0.25] * len(alt_C), alt_C * right_scaling, right_cmap, right_norm, lw=4
            )
        


        # Draw a horizontal line indicating the duration of each period for the corresponding year, from "start" to "end" date.
        for i in periods.index:
            y = get_net_year(periods.loc[i, "start"])
            # Transpose period start and end to the 2014-2015 winter, then convert using mdates.date2num.
            start = periods.loc[i, "start"]
            start = mdates.date2num(
                dt.datetime(2014 + (start.year - y), start.month, start.day)
            )
            end = periods.loc[i, "end"]
            end = mdates.date2num(dt.datetime(2014 + (end.year - y), end.month, end.day))
            ax.plot([start, end], [y - 0.12, y - 0.12], c="k", lw=3)
        
        # Do the same for the alternative periods
        if alt_periods is not None:
            for i in alt_periods.index:
                y = get_net_year(alt_periods.loc[i, "start"])
                # Transpose period start and end to the 2014-2015 winter, then convert using mdates.date2num.
                start = alt_periods.loc[i, "start"]
                start = mdates.date2num(
                    dt.datetime(2014 + (start.year - y), start.month, start.day)
                )
                end = alt_periods.loc[i, "end"]
                end = mdates.date2num(dt.datetime(2014 + (end.year - y), end.month, end.day))
                ax.plot([start, end], [y + 0.12, y + 0.12], c="royalblue", lw=3)

        ax.set_xlim(dates[0], dates[-1])
        ax.set_ylim(year_selection[0] - 0.5, year_selection[-1] + 0.5)

        # Vertically flip y-axis
        ax.invert_yaxis()

        # Display all years as ticks.
        ax.set_yticks(year_selection)

        # Format each y tick as 2019/20 instead of 2019 (for example)
        ax.set_yticklabels([f"{str(y)[2:]}/{str(y + 1)[2:]}" for y in year_selection])

        # Format x ticks as month names.
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

        # Remove tick marks
        ax.tick_params(axis="y", which="both", length=0)

        # Turn off axis frame
        ax.set_frame_on(False)

    # Add a colorbar.
    cax = fig.add_axes([0.15, 0, 0.3, 0.02])
    cb = mpl.colorbar.ColorbarBase(cax, cmap=left_cmap, norm=left_norm, orientation="horizontal")
    cb.set_label(left_str)
    cb.set_ticks(left_ticks)
    cb.set_ticklabels([str(t) for t in left_ticks])
    cax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    cax.set_frame_on(False)

    # Add a colorbar.
    cax = fig.add_axes([0.57, 0, 0.3, 0.02])
    cb = mpl.colorbar.ColorbarBase(cax, cmap=right_cmap, norm=right_norm, orientation="horizontal")
    cb.set_label(right_str)
    cb.set_ticks(right_ticks)
    cb.set_ticklabels([str(t) for t in right_ticks])
    cax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    cax.set_frame_on(False)


    axs[1].plot([], [], c="k", lw=2, label="S.d. events")
    if alt_periods is not None:
        axs[1].plot([], [], c="royalblue", lw=2, label="Alternative events")

    # Place legend to the right of the colour bar.
    axs[1].legend(loc="center left", bbox_to_anchor=(-0.5, -0.35), frameon=False)

    if save_fig:
        if path_str is None:
            print("Please provide a name for the saved figure.")
        else:
            fig.savefig(f"{path_str}.pdf", bbox_inches='tight')
    else:
        plt.show(); 
    return fig, axs

def plot_scatter(ax, x_data, y_data, x_label, y_label, title):
    ax.scatter(x_data, y_data, s=1)
    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)
    ax.set_title(title, fontsize=12)
    corr = x_data.corr(y_data)
    ax.annotate(f"Corr: {corr:.2f}", xy=(0.95, 0.95), xycoords='axes fraction', ha='right', va='top', fontsize=10)

def plot_generation_stack(
    n: pypsa.Network,
    start: pd.Timestamp,
    end: pd.Timestamp,
    difficult_periods: pd.DataFrame,
    freq: str = "1D",
    new_index: pd.TimedeltaIndex = None,
    ax=None,
):
    """Plot the generation stack with highlighted difficult periods."""
    if ax is None:
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot()

    # Gather generation, storage, and load data.
    # NB: Transmission cancels out on the European level.
    p_gen = n.generators_t.p.groupby(n.generators.carrier, axis=1).sum() / 1e3
    p_store = n.stores_t.p.groupby(n.stores.carrier, axis=1).sum() / 1e3
    p_storage = n.storage_units_t.p.groupby(n.storage_units.carrier, axis=1).sum() / 1e3

    p = pd.concat([p_gen, p_store, p_storage], axis=1)
    p = p.resample(freq).mean()
    # Ensure we have no leap days.
    p = p[~((p.index.month == 2) & (p.index.day == 29))]
    p_neg = p.clip(upper=0)
    p = p.clip(lower=0)

    loads = n.loads_t.p_set.sum(axis=1).resample(freq).mean() / 1e3
    # Ensure we have no leap days.
    loads = loads[~((loads.index.month == 2) & (loads.index.day == 29))]

    # Ensure load shedding has a colour.
    n.carriers.color["load-shedding"] = "#000000"
    colors = [n.carriers.color[carrier] for carrier in p.columns]

    if new_index is not None:
        # Reindex the dates, as the years are wrong.
        p.index = new_index
        p_neg.index = new_index
        loads.index = new_index

    # If we specified a start and end date, only plot that period.
    p_slice = p.loc[start:end]
    p_neg_slice = p_neg.loc[start:end]
    loads = loads[start:end]

    # Plot the generation stack.
    ax.stackplot(p_slice.index, p_slice.transpose(), colors=colors, labels=p.columns)
    ax.stackplot(p_neg_slice.index, p_neg_slice.transpose(), colors=colors)

    # Plot the difficult periods.
    # First the ones we have identified.
    ymin, ymax = ax.get_ylim()
    for _, row in difficult_periods.iterrows():
        if (
            row["start"].tz_localize(None) > start
            and row["end"].tz_localize(None) < end
        ):
            period_start = pd.Timestamp(row["start"].date())
            period_end = pd.Timestamp(row["end"].date())
            ax.fill_between(
                pd.DatetimeIndex(pd.date_range(period_start, period_end, freq="D")),
                ymin,
                ymax,
                color="grey",
                alpha=0.3,
                label="Filtered period",
            )
    # Load.
    ax.plot(loads, ls="dashed", color="red", label="load", linewidth=0.5)
    ax.set_xlim(start, end)

    if ax is None:
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    ax.set_ylabel("GW")
    plt.tight_layout()


def plot_cluster_anomalies(
    flex_anomaly: pd.DataFrame,
    system_anomaly: pd.DataFrame,
    periods: pd.DataFrame,
    cluster_nr: int,
    clustered_vals: pd.DataFrame,
    tech: list = ["biomass", "nuclear", "H2 fuel cell", "battery discharger", "PHS", "hydro"],
    tech_colours: list = ["#baa741", "#ff8c00", "#c251ae", "#ace37f", "#51dbcc", "#298c81"],
    plot_all_system: bool = True,
    resampled: str = "1D",
    save_fig: bool = False,
    path_str: str = None,
    cluster_names: list = None,
):
    for cluster in range(cluster_nr):
        nb_plots = len(clustered_vals.loc[clustered_vals["cluster"] == cluster])
        nb_rows = nb_plots // 4 if nb_plots % 4 == 0 else nb_plots // 4 + 1

        fig, axs = plt.subplots(nb_rows, 4, figsize = (30 * cm, nb_rows * 7 * cm), sharey=True, gridspec_kw={'hspace': 0.6})
        if len(cluster_names) == cluster_nr:
            fig.suptitle(f"Cluster {cluster}: {cluster_names[cluster]}", fontsize=16)
        else:
            fig.suptitle(f"Cluster {cluster}", fontsize=16)

        for i, event_nr in enumerate(clustered_vals[clustered_vals["cluster"] == cluster].index):
            ax = axs.flatten()[i]
            start = periods.loc[event_nr, "start"]
            end = periods.loc[event_nr, "end"]

            # Plot stack plot of flexibility.
            p = flex_anomaly.loc[start:end, tech].astype(float).resample(resampled).mean() / 1e3 # in GW
            p_neg = p.clip(upper=0)
            p_pos = p.clip(lower=0)

            ax.stackplot(p_pos.index, p_pos.T, colors=tech_colours, labels=p_pos.columns)
            ax.stackplot(p_neg.index, p_neg.T, colors=tech_colours)

            # Plot net load anomaly.
            net_load = system_anomaly.loc[start:end, "Net load anomaly"].resample(resampled).mean() / 1e3
            ax.plot(net_load.index, net_load, color="black", lw=1, ls="--", label="Net load anomaly")
            if plot_all_system:
                load = system_anomaly.loc[start:end, "Load anomaly"].resample(resampled).mean() / 1e3
                wind = system_anomaly.loc[start:end, "Wind anomaly"].resample(resampled).mean() / 1e3
                solar = system_anomaly.loc[start:end, "Solar anomaly"].resample(resampled).mean() / 1e3
                ax.plot(load.index, load, color="red", lw=1, ls=":", label="Load anomaly")
                ax.plot(wind.index, wind, color="blue", lw=1, ls=":", label="Wind anomaly")
                ax.plot(solar.index, solar, color="grey", lw=1, ls=":", label="Solar anomaly")
            
            ax.set_title(f"Event {event_nr}")
            ax.set_ylabel("Flexibility/Anomaly [GW]")
            ax.set_xlabel(f"{start.date()} - {end.date()}", fontsize=8)

            # Set x-ticks to only show day and month
            ax.xaxis.set_major_locator(mdates.DayLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
            # Change font size of x tick labels.
            ax.tick_params(axis="x", labelsize=8, rotation=90)
        
        # Add legend to the last row between the 2nd and 3rd plot.
        labels, handles = ax.get_legend_handles_labels()
        axs.flatten()[-3].legend(
            labels, handles, loc="center", bbox_to_anchor=(1, -0.6), frameon=False, ncols=4
        )

        # Hide empty plots.
        for ax in axs.flatten()[nb_plots:]:
            ax.axis("off")

        if save_fig:
            if str is None:
                print("Please specify a path to save the figure.")
            else:
                plt.savefig(f"{path_str}_cluster_{cluster}.pdf", bbox_inches='tight')
        else:
            plt.show();


def plot_period_anomalies(
    flex_anomaly: pd.DataFrame,
    system_anomaly: pd.DataFrame,
    periods: pd.DataFrame,
    tech: list = ["biomass", "nuclear", "H2 fuel cell", "battery discharger", "PHS", "hydro"],
    tech_colours: list = ["#baa741", "#ff8c00", "#c251ae", "#ace37f", "#51dbcc", "#298c81"],
    plot_all_system: bool = True,
    resampled: str = "1D",
    save_fig: bool = False,
    path_str: str = None,
):
    nb_plots = len(periods)
    nb_rows = nb_plots // 4 if nb_plots % 4 == 0 else nb_plots // 4 + 1

    fig, axs = plt.subplots(nb_rows, 4, figsize = (30 * cm, nb_rows * 7 * cm), sharey=True, gridspec_kw={'hspace': 0.6})

    for i, row in periods.iterrows():
        ax = axs.flatten()[i]
        start = row["start"]
        end = row["end"]

        # Plot stack plot of flexibility.
        p = flex_anomaly.loc[start:end, tech].astype(float).resample(resampled).mean() / 1e3 # in GW
        p_neg = p.clip(upper=0)
        p_pos = p.clip(lower=0)

        ax.stackplot(p_pos.index, p_pos.T, colors=tech_colours, labels=p_pos.columns)
        ax.stackplot(p_neg.index, p_neg.T, colors=tech_colours)

        # Plot net load anomaly.
        net_load = system_anomaly.loc[start:end, "Net load anomaly"].resample(resampled).mean() / 1e3
        ax.plot(net_load.index, net_load, color="black", lw=1, ls="--", label="Net load anomaly")
        if plot_all_system:
            load = system_anomaly.loc[start:end, "Load anomaly"].resample(resampled).mean() / 1e3
            wind = system_anomaly.loc[start:end, "Wind anomaly"].resample(resampled).mean() / 1e3
            solar = system_anomaly.loc[start:end, "Solar anomaly"].resample(resampled).mean() / 1e3
            ax.plot(load.index, load, color="red", lw=1, ls=":", label="Load anomaly")
            ax.plot(wind.index, wind, color="blue", lw=1, ls=":", label="Wind anomaly")
            ax.plot(solar.index, solar, color="grey", lw=1, ls=":", label="Solar anomaly")
        
        ax.set_title(f"Event {i}")
        ax.set_ylabel("Flexibility/Anomaly [GW]")
        ax.set_xlabel(f"{start.date()} - {end.date()}", fontsize=8)

        # Set x-ticks to only show day and month
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
        # Change font size of x tick labels.
        ax.tick_params(axis="x", labelsize=8, rotation=90)
    
    # Add legend to the last row between the 2nd and 3rd plot.
    labels, handles = ax.get_legend_handles_labels()
    axs.flatten()[-3].legend(
        labels, handles, loc="center", bbox_to_anchor=(1, -0.6), frameon=False, ncols=4
    )

    # Hide empty plots.
    for ax in axs.flatten()[nb_plots:]:
        ax.axis("off")

    if save_fig:
        if str is None:
            print("Please specify a path to save the figure.")
        else:
            plt.savefig(f"{path_str}.pdf", bbox_inches='tight')
    else:
        plt.show();


colours = {
    "DC":  "#8a1caf",
    "AC": "#70af1d",
    "biomass": "#baa741",
    "nuclear": "#ff8c00",
    "ror": "#3dbfb0",
    "fuel_cells": "#c251ae",
    "battery": "#ace37f",
    "phs": "#51dbcc",
    "hydro": "#298c81",
}


def plot_flex_events(
    periods: pd.DataFrame,
    all_flex: pd.DataFrame,
    avg_flex: pd.DataFrame,
    rolling: int = 24,
    mark_sde: bool = True,
    window_length: pd.Timedelta = pd.Timedelta("30d"),
    tech: list = ['DC', 'AC', 'biomass', 'nuclear', 'ror', 'fuel_cells','battery', 'phs','hydro'],
    title: str = "Flexibility usage",
    save_fig: bool = False,
    path_str: str = None,
):
    colours = {
    "DC":  "#8a1caf",
    "AC": "#70af1d",
    "biomass": "#baa741",
    "nuclear": "#ff8c00",
    "ror": "#3dbfb0",
    "fuel_cells": "#c251ae",
    "battery": "#ace37f",
    "phs": "#51dbcc",
    "hydro": "#298c81",
    }
    
    nrows = len(periods) // 4 if len(periods) % 4 == 0 else len(periods) // 4 + 1
    fig, axs = plt.subplots(nrows=nrows, ncols = 4, figsize=(30 * cm, 50 * cm), sharey=True, gridspec_kw={'hspace': 0.5})
    fig.suptitle(title, fontsize=12)
    # No vertical space between title and the rest of the plot.
    fig.subplots_adjust(top=0.95)

    for i, row in periods.iterrows():
        window_start = row["start"] - window_length
        window_end = row["end"] + window_length

        ax = axs.flatten()[i]
        ax.plot(
            all_flex.loc[window_start:window_end, tech].rolling(rolling).mean().index,
            all_flex.loc[window_start:window_end, tech].rolling(rolling).mean(),
            color = [colours[t] for t in tech] if len(tech) > 1 else colours[tech[0]],
            lw=0.5,
        )

        # Translated 2014-2015 from avg_flex to correct year.
        shifted_avg_flex = avg_flex.copy()
        year = get_net_year(row["start"])
        new_index = pd.date_range(f"{year}-07-01", f"{year + 1}-06-30 23:00:00", freq="h")
        if pd.Timestamp(f"{year + 1}-01-01").is_leap_year:
            new_index = new_index.drop(pd.date_range(f"{year + 1}-02-29", f"{year + 1}-02-29 23:00:00", freq="h"))
        shifted_avg_flex.index = new_index


        ax.plot(
            shifted_avg_flex.loc[window_start:window_end, tech].rolling(rolling).mean().index,
            shifted_avg_flex.loc[window_start:window_end, tech].rolling(rolling).mean(),
            color = [colours[t] for t in tech] if len(tech) > 1 else colours[tech[0]],
            lw=0.25,
            ls = "dashed",
        )
        if mark_sde:
            ax.fill_between(
                all_flex.loc[row["start"]:row["end"], tech].rolling(rolling).mean().index,
                all_flex.loc[window_start:window_end, tech].rolling(rolling).mean().min(),
                all_flex.loc[window_start:window_end, tech].rolling(rolling).mean().max(),
                color="gray",
                alpha=0.2,
            )

        ax.set_title(f"Event {i}")
        #ax.set_ylim(0, 1)
        ax.set_xlim(window_start, window_end)
        
        # Only mark beginning of months in x-tickmarkers.
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%Y"))


        #ax.set_xlabel("Time")
        ax.set_ylabel("Flexibility usage")
    
    if save_fig:
        if path_str is None:
            print("Please specify a path to save the figure.")
        else:
            plt.savefig(f"{path_str}.pdf", bbox_inches='tight')
    else:
        plt.show();
    

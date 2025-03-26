# SPDX-FileCopyrightText: 2025 Aleksander Grochowicz
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Process the load shedding data from the validation when operating the optimal networks.
"""

import pandas as pd
import numpy as np
import pypsa
import os
import yaml

if __name__ == "__main__":
    process_network = False

    # Following the load_opt_networks function in notebook_utilities, without loading the networks.
    config_name = "stressful-weather"
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


    for design_year in years:
        ls_ts = []
        for operational_year in years:
            if operational_year == design_year:
                continue
            else:
                if process_network:
                    n = pypsa.Network(f"../results/stressful-weather/weather_year_{design_year}/networks/weather_year_{operational_year}_base_s_90_elec_lc1.25_Co2L.nc")
                    load_shedding = n.generators_t.p.filter(like="load", axis="columns").round(0)
                    load_shedding = load_shedding.loc[:,~load_shedding.columns.str.contains("battery|H2")]
                    fold = f"../results/stressful-weather/weather_year_{design_year}/validation/"
                    os.makedirs(fold, exist_ok=True)
                    load_shedding.to_csv(f"{fold}/{operational_year}_base_s_90_elec_lc1.25_Co2L.csv")
                else:
                    ls_ts.append(pd.read_csv(f"../results/stressful-weather/weather_year_{design_year}/validation/weather_year_{operational_year}_base_s_90_elec_lc1.25_Co2L.csv", index_col=0))
            
        nodal_shedding = pd.concat(ls_ts, axis="index", keys = [y for y in years if y != design_year])
        system_shedding = nodal_shedding.mean(axis="columns")

        mean_shedding = (sum(ls_ts)/len(ls_ts))
        mean_system_shedding = mean_shedding.mean(axis="columns")

        # Look at maximal average load shedding per node.
        max_shedding = mean_shedding.max(axis="index")
        # Look at expected nodal load shedding per year.
        expected_shedding = mean_shedding.sum(axis="index")


        # Export to CSV

        # Make sure folders exist.
        os.makedirs(f"./load_shedding/{config_name}/design_years/{design_year}/", exist_ok=True)
        nodal_shedding.to_csv(f"./load_shedding/{config_name}/design_years/{design_year}/nodal_shedding.csv")
        system_shedding.to_csv(f"./load_shedding/{config_name}/design_years/{design_year}/system_shedding.csv")
        mean_shedding.to_csv(f"./load_shedding/{config_name}/design_years/{design_year}/mean_nodal_shedding.csv")
        mean_system_shedding.to_csv(f"./load_shedding/{config_name}/design_years/{design_year}/mean_system_shedding.csv")
        max_shedding.to_csv(f"./load_shedding/{config_name}/design_years/{design_year}/max_shedding.csv")
        expected_shedding.to_csv(f"./load_shedding/{config_name}/design_years/{design_year}/expected_shedding.csv")

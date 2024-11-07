# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2023-2024 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT

# Based on: https://gist.github.com/fneum/47b857862dd9148a22eca5a2e85caa9a

if "snakemake" in globals():
    filename = snakemake.output[0]
else:
    filename = "weather_scenarios.yaml"

import itertools

template = """
weather_year_{year}:
  snapshots:
    start: "{year}-07-01 00:00"
    end: "{end_year}-06-30 23:00"
    inclusive: "both"
  atlite:
    default_cutout: europe-era5_1980-2020
    cutouts:
      europe-era5_1980-2020:
        module: era5
        x: [-12., 42.]
        y: [33., 72]
        dx: 0.3
        dy: 0.3
        time: ['{year}', '{end_year}']
  renewable:
    onwind:
      cutout: europe-era5_1980-2020
    offwind-ac:
      cutout: europe-era5_1980-2020
    offwind-dc:
      cutout: europe-era5_1980-2020
    solar:
      cutout: europe-era5_1980-2020
    hydro:
      cutout: europe-era5_1980-2020
  solar_thermal:
    cutout: europe-era5_1980-2020
  sector:
    heat_demand_cutout: europe-era5_1980-2020
"""

config_values = dict(year=range(1980, 2020), end_year = range(1981, 2021))

combinations = [
    dict(zip(config_values.keys(), values))
    for values in itertools.product(*config_values.values())
    if values[0] +1 == values[1]
]

with open(filename, "w") as f:
    for i, config in enumerate(combinations):
        f.write(template.format(scenario_number=i, **config))
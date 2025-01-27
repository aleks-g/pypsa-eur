# SPDX-FileCopyrightText: 2024 Aleksander Grochowicz & Koen van Greevenbroek
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Compute mean wind and solar capacity factors and loads."""

import numpy as np
import pandas as pd
import pypsa

if __name__ == "__main__":
    # Load input networks
    ns = [pypsa.Network(i) for i in snakemake.input]

    ns_wind_cf = []
    ns_solar_cf = []

    # Collect all generators, as the clustering might have led to different network layouts.
    for n in ns:
        wind_i = n.generators.index[
            n.generators.carrier.isin(["onwind", "offwind-ac", "offwind-dc", "offwind-float"])
        ]
        solar_i = n.generators.index[n.generators.carrier.isin(["solar", "solar-hsat"])]

        ns_wind_cf.append(n.generators_t.p_max_pu.loc[:, wind_i])
        ns_solar_cf.append(n.generators_t.p_max_pu.loc[:, solar_i])

    # Only take the intersection of all generators across the networks.
    wind_i = list(set.intersection(*[set(df.columns) for df in ns_wind_cf]))
    solar_i = list(set.intersection(*[set(df.columns) for df in ns_solar_cf]))


    # Compute mean wind capacity factors over all years
    wind_cf = pd.DataFrame(
        sum(
            [df.loc[:, wind_i].values for df in ns_wind_cf],
        )
        / len(ns),
        columns=wind_i,
        index=ns[0].snapshots,
    )

    # Periodic fit using fourier series.
    wind_cf_mean = pd.DataFrame(
        (
            np.fft.irfft(np.fft.rfft(wind_cf[col])[:3], len(wind_cf.index))
            for col in wind_cf.columns
        ),
        columns=wind_cf.index,
        index=wind_cf.columns,
    ).T
    wind_cf_mean.to_csv(snakemake.output.wind)

    # Compute mean solar capacity factors over all years
    solar_cf = pd.DataFrame(
        sum(
            [df.loc[:, solar_i].values for df in ns_solar_cf],
        )
        / len(ns),
        columns=solar_i,
        index=ns[0].snapshots,
    )

    # Periodic fit using fourier series.
    solar_cf_mean = pd.DataFrame(
        (
            np.fft.irfft(np.fft.rfft(solar_cf[col])[:3], len(solar_cf.index))
            for col in solar_cf.columns
        ),
        columns=solar_cf.index,
        index=solar_cf.columns,
    ).T
    solar_cf_mean.to_csv(snakemake.output.solar)

    # Now for load
    load = pd.DataFrame(
        sum(
            [n.loads_t.p_set.values for n in ns],
        )
        / len(ns),
        columns=ns[0].loads_t.p_set.columns,
        index=ns[0].loads_t.p_set.index,
    )

    # Periodic fit using fourier series.
    load_mean = pd.DataFrame(
        (
            np.fft.irfft(np.fft.rfft(load[col])[:3], len(load.index))
            for col in load.columns
        ),
        columns=load.index,
        index=load.columns,
    ).T
    load_mean.to_csv(snakemake.output.load)

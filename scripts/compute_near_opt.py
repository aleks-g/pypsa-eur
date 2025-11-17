# SPDX-FileCopyrightText: 2025 Aleksander Grochowicz
#
# SPDX-License-Identifier: MIT

"""
Inspired by https://github.com/koen-vg/enabling-agency/blob/main/workflow/scripts/compute_near_opt.py."""

from __future__ import annotations

import hashlib
import json
import logging
import signal
import tempfile
from multiprocessing import get_context
from pathlib import Path
from typing import TYPE_CHECKING, Any
import pypsa

from _helpers import (
    configure_logging,
    set_scenario_config,
    update_config_from_wildcards,
)
from solve_second_network import fix_networks

import linopy
import numpy as np
import pandas as pd
from linopy import LinearExpression, QuadraticExpression, merge
from scipy.stats.qmc import Halton

from pypsa._options import options
from pypsa.descriptors import nominal_attrs
from pypsa.optimization.mga import (
    generate_directions_halton,
    generate_directions_random,
    _convert_to_dict,
    _worker_init,
    OptimizationAbstractMGAMixin,
)

from scripts._helpers import PyPSA_V1

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pypsa import Network
logger = logging.getLogger(__name__)

def load_mga_dimensions(n, config_file='projection.yaml'):
      import yaml

      with open(config_file) as f:
          config = yaml.safe_load(f)

      dimensions = {}
      for category, specs in config['projection'].items():
          for spec in specs:
              carrier = spec['carrier']
              component = spec['component']
              attribute = spec['attribute']
              weight_attr = spec['weight']

              comp_df = n.c[component].static
              matching = comp_df[comp_df['carrier'] == carrier].index

              if len(matching) > 0:
                  dimensions[carrier] = {
                      component: {
                          attribute: {g: comp_df.loc[g, weight_attr] for g in matching}
                      }
                  }

      return dimensions

def mga_minmax(
    n: Network,
    dimensions: dict,
    snapshots: Sequence | None = None,
    multi_investment_periods: bool = False,
    slack: float = 0.05,
    model_kwargs: dict | None = None,
    max_parallel: int = 4,
    cache_dir: str | None = None,
    **kwargs: Any,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Perform MGA optimization using min-max directions. TODO: docstring"""
    # Check if minmax runs already exist. Reuse them in that case.
    mga_directions = list(np.eye(len(dimensions))) + list(-np.eye(len(dimensions)))
    dim_names = list(dimensions.keys())
    mga_names = ["min_" + c for c in dim_names] + ["max_" + c for c in dim_names]
    directions_df, coordinates_df, caps_df, last_iter = reuse_results(
        cache_dir=cache_dir,
        cache_key="minmax",
    )
    if last_iter == len(mga_directions):
        # TODO: Add check that directions match.
        logger.info("Min-max directions already computed. Reusing cached results.")
        return directions_df, coordinates_df, caps_df
    else:
        logger.info("Computing min-max directions for MGA.")
        # Check which directions are missing.
        remaining_directions = []
        remaining_names = []
        for i, direction in enumerate(mga_directions):
            if directions_df is not None:
                dir_rounded = np.round(direction, decimals=3)
                dirs_rounded = directions_df.round(decimals=3).to_numpy()
                if any(np.all(dir_rounded == dr, axis=0) for dr in dirs_rounded):
                    continue
            remaining_directions.append(direction)
            remaining_names.append(mga_names[i])
        # Run MGA for remaining directions.
        if remaining_directions:
            successful_directions, successful_coordinates, successful_caps = (
                n.optimize.optimize_mga_in_multiple_directions_cache(
                    directions=remaining_directions,
                    dimensions=dimensions,
                    cache_key="minmax",
                    cache_dir=cache_dir,
                    snapshots=snapshots,
                    multi_investment_periods=multi_investment_periods,
                    slack=slack,
                    model_kwargs=model_kwargs,
                    max_parallel=max_parallel,
                    **kwargs,
                )
            )
            # Assign names to successful directions.
            successful_directions.index = remaining_names
            successful_coordinates.index = remaining_names
            successful_caps.index = remaining_names
            # Combine with existing results.
            if directions_df is not None and not directions_df.empty:
                directions_df = pd.concat(
                    [directions_df, successful_directions], ignore_index=False
                )
                coordinates_df = pd.concat(
                    [coordinates_df, successful_coordinates], ignore_index=False
                )
                caps_df = pd.concat(
                    [caps_df, successful_caps], ignore_index=False
                )
            else:
                directions_df = successful_directions
                coordinates_df = successful_coordinates
                caps_df = successful_caps
        return directions_df, coordinates_df, caps_df


# TODO: Until https://github.com/aleks-g/PyPSA/blob/mga-caching/pypsa/optimization/mga.py is merged into main PyPSA we add the edits here.

def hash_mga(  # noqa: ANN201
    n: Network,
    directions: pd.DataFrame,
    dimensions: dict,
    slack: float = 0.05,
):
    """
    Hashes the MGA optimization problem for a given network and directions.

    Parameters
    ----------
    n : pypsa.Network
        The network to optimize.
    directions : pd.DataFrame
        DataFrame containing the directions for the MGA.
    dimensions : dict
        Dictionary defining the dimensions of the optimization problem.
    slack : float, optional
        Slack value for the optimization, by default 0.05.

    """
    hash_input = {
        "network_name": n._name,
        "network_meta": n._meta,
        "directions": directions.to_dict(orient="records"),
        "dimensions": dimensions,
        "slack": slack,
    }

    hash_str = json.dumps(hash_input, sort_keys=True).encode("utf-8")
    hash_value = hashlib.sha256(hash_str).hexdigest()[:8]
    logger.info("Hash value for MGA optimization: %s", hash_value)
    return hash_value

def read_capacities(  # noqa: ANN201
    n: Network,
) -> pd.DataFrame:
    """Read optimal capacities from the network into a DataFrame."""
    caps = []
    attr_map = {
        "Line": "s_nom_opt",
        "Link": "p_nom_opt",
        "Generator": "p_nom_opt",
        "StorageUnit": "p_nom_opt",
        "Store": "e_nom_opt",
    }
    for comp, attr in attr_map.items():
        if PyPSA_V1:
            df = n.components[comp].static
        else:
            df = n.static(comp)
        if attr in df.columns:
            for name, value in df[attr].round(1).items():
                caps.append({"component": comp, "name": name, "attribute": attr, "capacity": value})
    return pd.DataFrame(caps)



def reuse_results(  # noqa: ANN201
    cache_dir: str,
    cache_key: str,
):
    """Check if cached results exist for a given cache key and load them if they exist."""
    cache_file_directions = Path(cache_dir) / f"directions_{cache_key}.csv"
    cache_file_coords = Path(cache_dir) / f"coords_{cache_key}.csv"
    cache_file_caps = Path(cache_dir) / f"p_nom_opt_{cache_key}.csv"
    if cache_file_directions.exists() and cache_file_coords.exists():
        # Load cached results
        directions_df = pd.read_csv(cache_file_directions)
        coords_df = pd.read_csv(cache_file_coords)
        caps_df = pd.read_csv(cache_file_caps)
        # Check if lengths match.
        if len(directions_df) != len(coords_df):
            msg = "Cached directions and coordinates lengths do not match."
            raise ValueError(msg)
        if len(directions_df) != len(caps_df):
            msg = "Cached directions and capacities lengths do not match."
            raise ValueError(msg)
        iter_nb = len(directions_df)
        if iter_nb > 0:
            logger.info("Found cached results for %s directions.", iter_nb)
        # Drop duplicates from directions_df and coords_df.
        directions_df = directions_df.drop_duplicates().reset_index(drop=True)
        coords_df = coords_df.drop_duplicates().reset_index(drop=True)
        caps_df = caps_df.drop_duplicates().reset_index(drop=True)
        last_iter = directions_df.index[-1] if not directions_df.empty else -1
        return directions_df, coords_df, caps_df, last_iter
    else:
        logger.info("No cached results found to re-use.")
        return None, None, None, -1

# TODO: Need to adapt Koen's function to also output `p_nom_opt` and also the network so we can keep analysing it later.
class OptimizationAbstractMGAMixin_cache(OptimizationAbstractMGAMixin):


    """Add caching functionality to OptimizationAbstractMGAMixin."""

    _n: Network

    @staticmethod
    def _solve_single_direction(
        fn: str,
        direction: dict,
        dimensions: dict,
        cache_key: str,
        cache_dir: str,
        snapshots: Sequence,
        multi_investment_periods: bool,
        slack: float,
        model_kwargs: dict,
        kwargs: dict,
    ) -> tuple[dict, pd.Series | None, str | None]:
        """
        Solve a single direction for parallel execution (helper method).

        Caches the results in a temporary file.
        This wrapper is necessary since the network is read from a file in
        this case; also simplifies the return argument management.

        """
        from pypsa.networks import Network  # noqa: PLC0415

        try:
            n = Network(fn)
            _, _, coordinates = n.optimize.optimize_mga_in_direction(
                direction=direction,
                dimensions=dimensions,
                snapshots=snapshots,
                multi_investment_periods=multi_investment_periods,
                slack=slack,
                model_kwargs=model_kwargs,
                **kwargs,
            )
        except KeyboardInterrupt:
            # Handle interruption gracefully
            logger.info("Worker process interrupted")
            return (direction, None, None)
        except Exception as e:
            # Log error but don't crash the worker
            logger.warning(
                "Error solving in direction",
                extra={"direction": direction, "error": str(e)},
            )
            return (direction, None, None)
        else:
            if cache_dir is not None and cache_key is not None:
                # saves results to cache
                Path(cache_dir).mkdir(parents=True, exist_ok=True)
                cache_directions_path = Path(cache_dir) / f"directions_{cache_key}.csv"
                cache_coordinates_path = Path(cache_dir) / f"coords_{cache_key}.csv"
                cache_caps_path = Path(cache_dir) / f"p_nom_opt_{cache_key}.csv"
                if cache_directions_path.exists():
                    cache_directions = pd.read_csv(cache_directions_path)
                else:
                    cache_directions = pd.DataFrame()
                if cache_coordinates_path.exists():
                    cache_coordinates = pd.read_csv(cache_coordinates_path)
                else:
                    cache_coordinates = pd.DataFrame()
                if cache_caps_path.exists():
                    cache_caps = pd.read_csv(cache_caps_path)
                else:
                    cache_caps = pd.DataFrame()
                # Append new results
                cache_directions = pd.concat(
                    [cache_directions, pd.DataFrame([direction])],
                    axis="index",
                    ignore_index=True,
                )
                cache_coordinates = pd.concat(
                    [cache_coordinates, pd.DataFrame([coordinates])],
                    axis="index",
                    ignore_index=True,
                )
                caps = read_capacities(n)
                caps_fn = f"caps_{Path(fn).stem}.csv"
                caps.to_csv(caps_fn, index=False)
                cache_caps = pd.concat(
                    [cache_caps, caps_fn],
                    axis="index",
                    ignore_index=True,
                )
                # Save back to CSV while preventing concurrent write issues
                cache_directions.to_csv(cache_directions_path, index=False)
                cache_coordinates.to_csv(cache_coordinates_path, index=False)
                cache_caps.to_csv(cache_caps_path, index=False)
            return (direction, coordinates, caps_fn)


    def optimize_mga_in_multiple_directions_cache(
        self,
        directions: list[dict] | pd.DataFrame,
        dimensions: dict,
        snapshots: Sequence | None = None,
        cache_dir: str | None = None,
        multi_investment_periods: bool = False,
        slack: float = 0.05,
        model_kwargs: dict | None = None,
        max_parallel: int = 4,
        **kwargs: Any,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Run MGA optimization in multiple directions in parallel, reusing existing results if available.

        This method performs modelling-to-generate-alternatives (MGA) optimization
        across multiple directions simultaneously using parallel processing. Each
        direction represents a different objective in the low-dimensional projection
        space defined by the dimensions parameter.

        Note that, in order to achieve parallelism, this method exports the network
        to a temporary NetCDF file which is then re-imported in each parallel process.
        This leads to a slight overhead in IO and disk space. The temporary file is
        always cleaned up after the optimization is complete, regardless of whether
        any errors occurred during the optimization.

        Parameters
        ----------
        directions : list[dict] | pd.DataFrame
            Multiple directions in the low-dimensional space. If a list, each element
            should be a dictionary with keys matching those in `dimensions` and values
            representing vector coordinates. If a DataFrame, rows represent directions
            and columns represent dimension names.
        dimensions : dict
            A dictionary representing the dimensions of the low-dimensional space.
            The keys are user-defined names for the dimensions (matching those in the
            `directions` argument), and the values are dictionaries with the same
            structure as the `weights` argument in `optimize_mga`.
        cache_dir : str | None, optional
            Directory to store cached results. If None, caching is disabled.
        snapshots : Sequence | None, optional
            Set of snapshots to consider in the optimization. If None, uses all
            snapshots from the network. Defaults to None.
        multi_investment_periods : bool, default False
            Whether to optimize as a single investment period or to optimize in
            multiple investment periods. Then, snapshots should be a ``pd.MultiIndex``.
        slack : float
            Cost slack for budget constraint. Defaults to 0.05.
        model_kwargs: dict
            Keyword arguments used by `linopy.Model`, such as `solver_dir` or
            `chunk`.
        max_parallel : int
            Maximum number of parallel processes to use for solving multiple directions.
            Defaults to 4.
        **kwargs:
            Keyword argument used by `linopy.Model.solve`, such as `solver_name`,

        Returns
        -------
        directions_df : pd.DataFrame
            DataFrame containing the successfully solved directions, where each row
            represents a direction and columns correspond to dimension names.
        coordinates_df : pd.DataFrame
            DataFrame containing the coordinates of each successfully solved network
            in the user-defined dimensions. Rows correspond to solved directions
            and columns to dimension names.
        caps_df : pd.DataFrame
            DataFrame containing the filenames for the optimal capacities of each successfully solved network.

        Examples
        --------
        >>> dimensions = {
        ...     "wind": {"Generator": {"p_nom": {"wind": 1}}},
        ...     "solar": {"Generator": {"p_nom": {"solar": 1}}}
        ... }
        >>> directions = pypsa.optimization.mga.generate_directions_random(["wind", "solar"], 10)
        >>> dirs_df, coords_df = n.optimize.optimize_mga_in_multiple_directions(
        ...     directions, dimensions, max_parallel=2
        ... )
        >>> dirs_df # doctest: +SKIP
                wind     solar
        0  0.958766  0.284198
        1 -0.937432 -0.348170
        2 -0.805652  0.592389
        ...
        >>> coords_df # doctest: +ELLIPSIS
        wind  solar
        0   0.0    0.0
        1   0.0    0.0
        2   0.0    0.0
        ...

        """
        if cache_dir is None:
            logger.info(
                "No cache directory provided. Running optimizations without caching."
            )
            directions, coordinates = self.optimize_mga_in_multiple_directions(
                directions=directions,
                dimensions=dimensions,
                snapshots=snapshots,
                multi_investment_periods=multi_investment_periods,
                slack=slack,
                model_kwargs=model_kwargs,
                max_parallel=max_parallel,
                **kwargs,
            )
            return (directions, coordinates, None)
        else:
            # Ensure cache directory exists
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            # Check if cached results already exist for these parameters.
            if cache_key is None:
                cache_key = hash_mga(self._n, directions, dimensions, slack)
            cached_directions, cached_coordinates, cached_caps, last_iter = reuse_results(
                cache_dir=cache_dir, cache_key=cache_key
            )
            if last_iter < 0:
                logger.info(
                    "No previous iterations found. Starting optimizations from scratch."
                )
                cached_directions = pd.DataFrame()
                cached_coordinates = pd.DataFrame()
                cached_caps = pd.DataFrame()
                if isinstance(directions, pd.DataFrame):
                    directions = list(directions.T.to_dict().values())
            else:
                logger.info("Found cached results for %s directions.", last_iter + 1)
                # First round the directions to avoid issues with floating point precision.
                directions_rounded = directions.round(decimals=3)
                cached_directions_rounded = cached_directions.round(decimals=3)
                # Remove rows from directions that are present in cached_directions
                filtered_directions = directions[
                    ~directions_rounded.apply(tuple, 1).isin(
                        cached_directions_rounded.apply(tuple, 1)
                    )
                ]
                cached_results = (cached_directions, cached_coordinates, cached_caps)
                if filtered_directions.empty:
                    logger.info(
                        "All directions already cached. Returning cached results."
                    )
                    return cached_results
                else:
                    logger.info(
                        "Continuing optimization from iteration %s with %s remaining directions.",
                        last_iter + 1,
                        len(filtered_directions),
                    )
                    if isinstance(filtered_directions, pd.DataFrame):
                        directions = list(filtered_directions.T.to_dict().values())
            successful_directions, successful_coordinates, successful_caps = (
                self.optimize_mga_with_cache(
                    directions=directions,
                    dimensions=dimensions,
                    cache_key=cache_key,
                    cache_dir=cache_dir,
                    snapshots=snapshots,
                    multi_investment_periods=multi_investment_periods,
                    slack=slack,
                    model_kwargs=model_kwargs,
                    max_parallel=max_parallel,
                    **kwargs,
                )
            )
            # Combine cached results with new results
            combined_directions = pd.concat(
                [cached_directions, successful_directions], ignore_index=True
            )
            combined_coordinates = pd.concat(
                [cached_coordinates, successful_coordinates], ignore_index=True
            )
            combined_caps = pd.concat(
                [cached_caps, successful_caps], ignore_index=True
            )
            return combined_directions, combined_coordinates, combined_caps

    def optimize_mga_with_cache(
        self,
        directions: pd.DataFrame,
        dimensions: dict,
        cache_key: str | None = None,
        cache_dir: str | None = None,
        snapshots: Sequence | None = None,
        multi_investment_periods: bool = False,
        slack: float = 0.05,
        model_kwargs: dict | None = None,
        max_parallel: int = 4,
        **kwargs,  # noqa: ANN003
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Optimize MGA with caching."""
        if isinstance(directions, pd.DataFrame):
            directions = directions.T.to_dict()
        # Create temporary file to export the network. Note: cannot pass
        # the network as an argument directly since it is not picklable.
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
            fn = f.name
            # Wrap in try-finally to ensure the netcdf file is deleted
            # even if an error occurs
            try:
                self._n.export_to_netcdf(fn)
                # Use a process pool to solve in parallel
                with (
                    get_context("spawn").Pool(
                        processes=max_parallel,
                        initializer=_worker_init,
                        maxtasksperchild=1,  # Kill workers after each task to prevent memory leaks
                    ) as pool
                ):
                    try:
                        results = pool.starmap(
                            OptimizationAbstractMGAMixin_cache._solve_single_direction,
                            [
                                (
                                    fn,
                                    direction,
                                    dimensions,
                                    cache_key,
                                    cache_dir,
                                    snapshots,
                                    multi_investment_periods,
                                    slack,
                                    model_kwargs,
                                    kwargs,
                                )
                                for direction in directions
                            ],
                        )
                    except Exception:
                        # Terminate all workers if something goes wrong.
                        pool.terminate()
                        pool.join()
                        raise
                # Separate successful and failed results
                sucessful = [
                    (direction, coords, caps)
                    for direction, coords, caps in results
                    if coords is not None
                ]
                failed_count = len(results) - len(sucessful)
                if failed_count > 0:
                    logger.warning(
                        "%s out of %s optimizations failed", failed_count, len(results)
                    )
                if not sucessful:
                    return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
                successful_directions, successful_coordinates, successful_caps = zip(
                    *sucessful, strict=True
                )
                return (
                    pd.DataFrame(successful_directions),
                    pd.DataFrame(successful_coordinates),
                    pd.DataFrame(successful_caps),
                )
            # TODO: Comment out the removal of the temporary file because we are only cleaning those up in a later rule.
            finally:
                pass
            #     # Clean up temporary file
            #     if Path(fn).exists():
            #         Path(fn).unlink()



if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "compute_near-opt",
            opts="",
            clusters="5",
            configfiles="config/test/config.overnight.yaml",
            sector_opts="",
            planning_horizons="2030",
        )

    configure_logging(snakemake)
    set_scenario_config(snakemake)
    update_config_from_wildcards(snakemake.config, snakemake.wildcards)

    solve_opts = snakemake.params.solving["options"]

    # Load network and solving options.
    n = pypsa.Network(snakemake.input.network)
    m = n.copy()
    fix_networks(m, n)



    # Load configuration options.
    near_opt_params = snakemake.config.get("near_opt", {})
    if near_opt_params == {}:
        raise ValueError("No near-opt configuration found in config file.")
    dimensions = near_opt_params["projection"]
    if near_opt_params["slack"]["relative"] == True:
        slack = near_opt_params["slack"]["value"]
    else:
        # Resilience premium, absolute terms
        slack = near_opt_params["slack"]["value"] / n.objective

    
    if near_opt_params["approx"]["minmax"]:
        directions_mga, coordinates_mga, caps_mga = mga_minmax(
            m,
            dimensions,
            snapshots=None,
            multi_investment_periods=False,
            slack=slack,
            model_kwargs=None,
            max_parallel=near_opt_params["approx"].get("num_parallel_solvers", 2),
            cache_dir=near_opt_params.get("cache_dir", None),
        )

    # Direction generation
    if near_opt_params["approx"]["directions"] == "random-uniform":
        num_directions = near_opt_params["approx"]["iterations"]
        directions = generate_directions_random(
            keys = list(dimensions.keys()),
            n_directions = num_directions,
            seed = near_opt_params["approx"].get("seed", 123),
        )
    elif near_opt_params["approx"]["directions"] == "halton":
        num_directions = near_opt_params["approx"]["iterations"]
        directions = generate_directions_halton(
            keys = list(dimensions.keys()),
            n_directions = num_directions,
        )
    else:
        raise ValueError("Unknown direction generation method.")
    
    run_mga_cache = (
        OptimizationAbstractMGAMixin_cache.optimize_mga_in_multiple_directions_cache.__get__(
            m.optimize
        )
    )
    
    directions_df, coordinates_df, caps_df = run_mga_cache(
        directions=directions,
        dimensions=dimensions,
        snapshots=None,
        cache_dir=near_opt_params.get("cache_dir", None),
        multi_investment_periods=False,
        slack=slack,
        model_kwargs=None,
        max_parallel=near_opt_params["approx"].get("num_parallel_solvers", 2),
    )

    # Concatenate results
    if near_opt_params["approx"]["minmax"]:
        directions_df = pd.concat(
            [directions_mga, directions_df], ignore_index=True
        )
        coordinates_df = pd.concat(
            [coordinates_mga, coordinates_df], ignore_index=True
        )
        caps_df = pd.concat(
            [caps_mga, caps_df], ignore_index=True
        )
    # Export results
    directions_df.to_csv(snakemake.output.directions, index=False)
    coordinates_df.to_csv(snakemake.output.coordinates, index=False)
    caps_df.to_csv(snakemake.output.capacities, index=False)    

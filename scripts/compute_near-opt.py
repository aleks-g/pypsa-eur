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

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pypsa import Network
logger = logging.getLogger(__name__)


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



    # If no previous runs exist, generate minmax directions.


    # Make some easy-to-read descriptions for the directions.


    # Run MGA in multiple directions with caching.


    # Save networks for now.


    # Return directions, coordinates, optimal capacities.
    return directions_df, coordinates_df, p_nom_opts_df


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


def reuse_results(  # noqa: ANN201
    cache_dir: str,
    cache_key: str,
):
    """Check if cached results exist for a given cache key and load them if they exist."""
    cache_file_directions = Path(cache_dir) / f"directions_{cache_key}.csv"
    cache_file_coords = Path(cache_dir) / f"coords_{cache_key}.csv"
    if cache_file_directions.exists() and cache_file_coords.exists():
        # Load cached results
        directions_df = pd.read_csv(cache_file_directions)
        coords_df = pd.read_csv(cache_file_coords)
        # Check if lengths match.
        if len(directions_df) != len(coords_df):
            msg = "Cached directions and coordinates lengths do not match."
            raise ValueError(msg)
        iter_nb = len(directions_df)
        if iter_nb > 0:
            logger.info("Found cached results for %s directions.", iter_nb)
        # Drop duplicates from directions_df and coords_df.
        directions_df = directions_df.drop_duplicates().reset_index(drop=True)
        coords_df = coords_df.drop_duplicates().reset_index(drop=True)
        last_iter = directions_df.index[-1] if not directions_df.empty else -1
        return directions_df, coords_df, last_iter
    else:
        logger.info("No cached results found to re-use.")
        return None, None, -1

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
    ) -> tuple[dict, pd.Series | None]:
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
            return (direction, None)
        except Exception as e:
            # Log error but don't crash the worker
            logger.warning(
                "Error solving in direction",
                extra={"direction": direction, "error": str(e)},
            )
            return (direction, None)
        else:
            if cache_dir is not None and cache_key is not None:
                # saves results to cache
                Path(cache_dir).mkdir(parents=True, exist_ok=True)
                cache_directions_path = Path(cache_dir) / f"directions_{cache_key}.csv"
                cache_coordinates_path = Path(cache_dir) / f"coords_{cache_key}.csv"
                if cache_directions_path.exists():
                    cache_directions = pd.read_csv(cache_directions_path)
                else:
                    cache_directions = pd.DataFrame()
                if cache_coordinates_path.exists():
                    cache_coordinates = pd.read_csv(cache_coordinates_path)
                else:
                    cache_coordinates = pd.DataFrame()
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
                # Save back to CSV while preventing concurrent write issues
                cache_directions.to_csv(cache_directions_path, index=False)
                cache_coordinates.to_csv(cache_coordinates_path, index=False)
            return (direction, coordinates)


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
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
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
            return self.optimize_mga_in_multiple_directions(
                directions=directions,
                dimensions=dimensions,
                snapshots=snapshots,
                multi_investment_periods=multi_investment_periods,
                slack=slack,
                model_kwargs=model_kwargs,
                max_parallel=max_parallel,
                **kwargs,
            )
        else:
            # Ensure cache directory exists
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            # Check if cached results already exist for these parameters.
            cache_key = hash_mga(self._n, directions, dimensions, slack)
            cached_directions, cached_coordinates, last_iter = reuse_results(
                cache_dir=cache_dir, cache_key=cache_key
            )
            if last_iter < 0:
                logger.info(
                    "No previous iterations found. Starting optimizations from scratch."
                )
                cached_directions = pd.DataFrame()
                cached_coordinates = pd.DataFrame()
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
                cached_results = (cached_directions, cached_coordinates)
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
            successful_directions, successful_coordinates = (
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
            return combined_directions, combined_coordinates

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
            # Wrap in try-finally to ensure the temporary file is deleted
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
                            OptimizationAbstractMGAMixin._solve_single_direction,
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
                    (direction, coords)
                    for direction, coords in results
                    if coords is not None
                ]
                failed_count = len(results) - len(sucessful)
                if failed_count > 0:
                    logger.warning(
                        "%s out of %s optimizations failed", failed_count, len(results)
                    )
                if not sucessful:
                    return pd.DataFrame(), pd.DataFrame()
                successful_directions, successful_coordinates = zip(
                    *sucessful, strict=True
                )
                return (
                    pd.DataFrame(successful_directions),
                    pd.DataFrame(successful_coordinates),
                )
            finally:
                # Clean up temporary file
                if Path(fn).exists():
                    Path(fn).unlink()



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

    # Load logging etc.


    # Load network and solving options.
    # Ensure network is pre-optimized; only then we can use pypsa.optimize.mga.

    # Load configuration options.
    # Need resilience premium. Redefine slack based on this.

    # Need dimensions.

    # Need direction generation.

    # Check if previous iterations exist.
    # If so, reuse direction generation.
    # Check if results are validated.
    # Check if networks exist.

    # If minmax.
    # mga_minmax(
    #     n,
    #     dimensions,
    #     snapshots=None,
    #     multi_investment_periods=False,
    #     slack=snakemake.params.mga.get("slack", 0.05),
    #     model_kwargs=None,
    #     max_parallel=snakemake.params.mga.get("max_parallel", 4),
    #     cache_dir=snakemake.params.mga.get("cache_dir", None),
    # )

    # Run MGA.
    # n.optimize.optimize_mga_in_multiple_directions_cache()

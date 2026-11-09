"""Calculate the likelihood of all the theoretical patterns according to the specified merit functions."""

import multiprocessing
import os
from functools import partial
from pathlib import Path

from foamppkg import maximum_likelihood_estimator as mle
from foamppkg.pipeline.pipeline_config import config

def run(config):

    ###############################################################################
    data_out_dir = Path(f"{os.getcwd()}/meritvalues")
    Path(data_out_dir).mkdir(parents=True, exist_ok=True)

    file_path = Path(f"V_matrix/{config.star}_determinant_conditionNr.tsv")
    # remove file if it exists to avoid duplicate entries on successive runs
    if file_path.is_file():
        file_path.unlink()
    args = []
    observables = []

    # Resolve config.md_max_components. "auto" is the rank of the theoretical patterns:
    # one direction per free parameter, plus one for the rotation when it is optimised
    # per model during pattern construction (every path except a p-mode-only fit).
    max_components = config.md_max_components
    if max_components == "auto":
        rotation_is_fitted = config.combined_fit is True or config.puls_type == "gmode"
        max_components = len(config.free_parameters) + (1 if rotation_is_fitted else 0)
        config.logger.info(
            f"md_max_components='auto' resolved to {max_components} "
            f"({len(config.free_parameters)} free parameters"
            f"{' + fitted rotation' if rotation_is_fitted else ''})"
        )
    elif max_components is not None:
        max_components = int(max_components)

    if config.combined_fit is False:
        if config.puls_type == 'gmode':
            observations = config.observations
        elif config.puls_type == 'pmode':
            observations = config.observations_pmode
        for grid in config.grids:
            for method in config.pattern_methods:
                for merit_function in config.merit_functions:
                    for obs in config.observable_seismic:
                        if obs == "P" or obs == "dP":
                            observed_quantity = "period"
                        elif obs == "f":
                            observed_quantity = "frequency"

                        theory_path = f"{config.main_directory}/extracted_freqs/surface+{observed_quantity}_{config.star}_{grid}_{method}.hdf"
                        observables = [obs]
                        if config.observable_additional is not None:
                            observables += config.observable_additional
                        args.append((theory_path, observables, merit_function))

            with multiprocessing.Pool(config.nr_cpu) as p:
                func = partial(
                    mle.calculate_likelihood,
                    obs_path=observations,
                    star_name=config.star,
                    fixed_params=config.fixed_parameters,
                    grid_parameters=config.grid_parameters,
                    max_components=max_components,
                    sigma_theo_gmode=config.sigma_theo_gmode,
                    gmode_covariance=config.gmode_covariance,
                )
                p.starmap(func, args)

    elif config.combined_fit is True:
        print('Combined p-mode + g-mode fitting')
        for grid in config.grids:
            for method in config.pattern_methods:
                theory_path = f"{config.main_directory}/extracted_freqs/surface+combined_{config.star}_{grid}_{method}.hdf"
                args.append(theory_path)


        # In a combined fit the g-modes are always the period pattern and the p-modes the
        # frequencies, so observable_seismic is split by which of the two each entry describes
        # rather than looped over the way the single-pulsation-type branch above does.
        observables_gmode, observables_pmode = mle.combined_observables(config.observable_seismic)
        # Fail before any work is done if a merit function has no combined implementation.
        for merit_function in config.merit_functions:
            mle.combined_suffix(merit_function)

        with multiprocessing.Pool(config.nr_cpu) as p:
            for merit_function in config.merit_functions:
                func = partial(
                    mle.calculate_likelihood_combined,
                    observables_gmode=observables_gmode,
                    observables_pmode=observables_pmode,
                    obs_path_gmode=config.observations,
                    obs_path_pmode=config.observations_pmode,
                    star_name=config.star,
                    fixed_params=config.fixed_parameters,
                    grid_parameters=config.grid_parameters,
                    sigma_theo_pmode=config.sigma_theo_pmode,
                    max_components=max_components,
                    sigma_theo_gmode=config.sigma_theo_gmode,
                    gmode_covariance=config.gmode_covariance,
                    merit_function=merit_function,
                )
                p.map(func, args)


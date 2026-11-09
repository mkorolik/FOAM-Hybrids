from pathlib import Path
import pandas as pd

import glob
import logging
import multiprocessing
from functools import partial
import sys
from lmfit import Minimizer, Parameters

from . import support_functions as sf

from . import functions_for_mesa as ffm
from . import build_optimized_pattern_pmodes as bopp
from .pipeline.pipeline_config import config
from . import gmode_rotation_scaling as grs
from . import functions_for_gyre as ffg




logger = logging.getLogger("logger.ffg")

def all_freqs_from_stormfile(path, parameters):
    _, data = sf.read_hdf5(path)
    param_dict = sf.get_param_from_filename(path, parameters, values_as_float=True)
    for j in range(len(data["frequency"]) - 1, -1, -1):
        n_pg = data["radial-order"][j]
        degree = int(data['degree'][j])
        if abs(n_pg) < 10:
            n_pg = f"{sf.sign(n_pg)}00{abs(n_pg)}"
        elif abs(n_pg) < 100:
            n_pg = f"{sf.sign(n_pg)}0{abs(n_pg)}"
        param_dict.update({f"l{degree}_n_pg{n_pg}": data["frequency"][j]})
    
    return param_dict


def extract_frequency_grid(storm_files, parameters, output_file='storm_grid_test.hdf', nr_cpu=None, exclude_models=None):
    mp_list = multiprocessing.Manager().list()

    # Glob all the files (skipping those of excluded models), then iteratively send them to a pool of processors
    summary_files = ffm.exclude_models(glob.glob(storm_files), exclude_models)
    with multiprocessing.Pool(nr_cpu) as p:
        extract_func = partial(all_freqs_from_stormfile, parameters=parameters)
        dictionaries = p.imap(extract_func, summary_files)
        for new_row in dictionaries:
            # Fill the listProxy with dictionaries for each read file
            mp_list.append(new_row)

        df = pd.DataFrame(data=list(mp_list))
    
    # Sort the columns with frequencies by their radial order
    column_list = list(df.columns[: len(parameters)])
    column_list.extend(sorted(df.columns[len(parameters) :]))
    df = df.reindex(column_list, axis=1)

    # Generate the directory for the output file and write the file afterwards
    Path(Path(output_file).parent).mkdir(parents=True, exist_ok=True)
    df.to_hdf(path_or_buf=output_file, key="pulsation_grid", format="table", mode="w")

    return df


def storm_m1_grid_file(config, grid):
    """Grid summary of the m=+1 STORM frequencies, used with pmode_m_treatment='first-order'."""
    return f"grid_summary/stormGrid_{grid}_m1_rotp{config.storm_rotation_m1}.hdf"


def extract_storm_m0_grid(config, grid, broken_models):
    """
    Extract the m=0 STORM frequencies of the grid, from STORM_out/rotp<rotation>_m0/
    (all rotp*_m0/ directories if config.prots is a list of rotation rates).
    The _m0 suffix keeps the m=+1 output in STORM_out/rotp<rotation>_m1/ out of this grid.
    """
    if config.prots == config.rotation_gyre:
        output_file = f"grid_summary/stormGrid_{grid}_rot{config.rotation_gyre}.hdf"
        storm_files = f"{config.grid_parent_directory}/{grid}/STORM_out/rotp{config.rotation_gyre}_m0/*/*{config.subgrid}*.hdf5"
    elif type(config.prots) is list:
        output_file = f"grid_summary/stormGrid_{grid}_rots.hdf"
        storm_files = f"{config.grid_parent_directory}/{grid}/STORM_out/rotp*_m0/*/*{config.subgrid}*.hdf5"
    else:
        return

    if not Path(output_file).is_file():
        extract_frequency_grid(
            storm_files,
            output_file=output_file,
            parameters=["rotp"] + config.grid_parameters,
            nr_cpu=config.nr_cpu,
            exclude_models=broken_models,
        )
    else:
        config.logger.warning(f"file already existed: {output_file}")


def extract_storm_m1_grid(config, grid, broken_models):
    """
    Extract the m=+1 STORM frequencies of the grid, from STORM_out/rotp<storm_rotation_m1>_m1/.
    """
    output_file = storm_m1_grid_file(config, grid)
    if not Path(output_file).is_file():
        extract_frequency_grid(
            f"{config.grid_parent_directory}/{grid}/STORM_out/rotp{config.storm_rotation_m1}_m1/*/*{config.subgrid}*.hdf5",
            output_file=output_file,
            parameters=["rotp"] + config.grid_parameters,
            nr_cpu=config.nr_cpu,
            exclude_models=broken_models,
        )
    else:
        config.logger.warning(f"file already existed: {output_file}")


def construct_pmode_pattern(config, grid, pulsation_grid_file, method, puls_file, gmode_puls_file=None):
    """
    Build the theoretical p-mode pattern of every model, treating rotationally split
    multiplets according to config.pmode_m_treatment.
    """
    if Path(puls_file).is_file():
        config.logger.warning(f"file already existed: {puls_file}")
        return

    if config.pmode_m_treatment == "first-order":
        bopp.construct_theoretical_puls_pattern_first_order(
            pulsation_grid_file, storm_m1_grid_file(config, grid), config.storm_rotation_m1,
            config.observations_pmode, output_file=puls_file, grid_parameters=config.grid_parameters,
            rotation_source=config.pmode_rotation, gmode_rotation_file=gmode_puls_file,
        )
    else:
        obs_dataframe = pd.read_table(config.observations_pmode, sep=r"\s+", header=0, index_col="index")
        if "m" in obs_dataframe.columns and (obs_dataframe["m"] != 0).any():
            config.logger.warning(
                "pmode_m_treatment='centroid' compares every p-mode with the m=0 frequencies, but the "
                "p-mode observations contain modes with m != 0; give multiplet centroids instead, "
                "or set pmode_m_treatment='first-order'"
            )
        bopp.construct_theoretical_puls_pattern_by_degree(
            pulsation_grid_file,
            config.observations_pmode,
            config.pmode_pattern_method or method,
            output_file=puls_file,
            grid_parameters=config.grid_parameters,
            nr_cpu=config.nr_cpu,
        )


def pipe0(config=config):
    for grid in config.grids:
        output_file = f"grid_summary/surfaceGrid_{grid}.hdf"
        if not Path(output_file).is_file():
            ffm.extract_surface_grid(
                f"{config.grid_parent_directory}/{grid}/MESA_out/*/profiles/*{config.subgrid}*.data",
                output_file=output_file,
                parameters=config.grid_parameters,
                nr_cpu=config.nr_cpu,
                additional_observables=config.observable_additional,
            )
        else:
            config.logger.warning(f"file already existed: {output_file}")

        # Skip the pulsation output of profiles that could not be read for the surface grid
        broken_models = ffm.read_broken_models(output_file)

        if config.combined_fit is False:
            if config.puls_type == 'gmode':
                for mval in config.mvals:
                    output_file = f"grid_summary/pulsationGrid_{grid}_rot{config.rotation_gyre}_k{config.kval}m{mval}.hdf"
                    if not Path(output_file).is_file():
                        ffg.extract_frequency_grid(
                            f"{config.grid_parent_directory}/{grid}/GYRE_out/rot{config.rotation_gyre}_k{config.kval}m{mval}/*/*{config.subgrid}*.HDF",
                            output_file=output_file,
                            parameters=["rot"] + config.grid_parameters,
                            nr_cpu=config.nr_cpu,
                            exclude_models=broken_models,
                        )
                    else:
                        config.logger.warning(f"file already existed: {output_file}")

            if config.puls_type == 'pmode':
                print('FOAM is not meant for only p-mode fitting; this branch should only come up if you are comparing your combined fit to individual g-mode and p-mode fits')
                extract_storm_m0_grid(config, grid, broken_models)
                if config.pmode_m_treatment == "first-order":
                    extract_storm_m1_grid(config, grid, broken_models)
        elif config.combined_fit is True:
            for mval in config.mvals:
                output_file = f"grid_summary/pulsationGrid_{grid}_rot{config.rotation_gyre}_k{config.kval}m{mval}.hdf"
                if not Path(output_file).is_file():
                    ffg.extract_frequency_grid(
                        f"{config.grid_parent_directory}/{grid}/GYRE_out/rot{config.rotation_gyre}_k{config.kval}m{mval}/*/*{config.subgrid}*.HDF",
                        output_file=output_file,
                        parameters=["rot"] + config.grid_parameters,
                        nr_cpu=config.nr_cpu,
                        exclude_models=broken_models,
                    )
                else:
                    config.logger.warning(f"file already existed: {output_file}")

            extract_storm_m0_grid(config, grid, broken_models)
            if config.pmode_m_treatment == "first-order":
                extract_storm_m1_grid(config, grid, broken_models)


def asymptotic_objects_per_m(config):
    """
    The asymptotic objects for the TAR rescaling of the g-mode pattern of each m in config.mvals.
    Each m has its own Hough function index k = l - |m| (e.g. k=0 for dipole sectoral modes,
    k=1 for dipole zonal modes), so a single config.kval cannot be used for all of them.
    """
    return {
        mval: grs.Asymptotic(gyre_dir=config.gyre_dir, kval=config.lval - abs(mval), mval=mval)
        for mval in config.mvals
    }


def pipe1(config=config):
    # The required asymptotic class object
    # asymp_obj = None
    asymp_obj = grs.Asymptotic(gyre_dir=config.gyre_dir, kval=config.kval, mval=config.mval)

    # Construct the pulsation patterns according to the different methods for the extracted theoretical grids
    for grid in config.grids:
        surface = f"grid_summary/surfaceGrid_{grid}.hdf"
        # Methods to construct theoretical pulsation patterns
        for method in config.pattern_methods:
            # Whether periods or frequencies are used
            observed_quantities = []
            if ("P" in config.observable_seismic) or ("dP" in config.observable_seismic):
                observed_quantities.append("period")
            if "f" in config.observable_seismic:
                observed_quantities.append("frequency")


            if config.combined_fit is False:
                puls_files={}
                rotname = 'rot'
                for observed_quantity in observed_quantities:
                    puls_file = f"extracted_freqs/{observed_quantity}_{config.star}_{grid}_{method}.hdf"                 
                    if config.puls_type == 'gmode':
                        pulsation_grid_files = {
                            mval: f"grid_summary/pulsationGrid_{grid}_rot{config.rotation_gyre}_k{config.kval}m{mval}.hdf"
                            for mval in config.mvals
                        }
                        asymp_objects = asymptotic_objects_per_m(config)
                        if not Path(puls_file).is_file():
                                bopp.construct_theoretical_puls_pattern_joint_m(
                                    pulsation_grid_files, config.observations, method,
                                    pattern_starting_pulsation=config.pattern_starting_pulsation[observed_quantity],
                                    output_file=puls_file, asymp_objects=asymp_objects,
                                    estimated_rotation=config.rotation_gyre, grid_parameters=config.grid_parameters,
                                    nr_cpu=config.nr_cpu,
                                )
                        else:
                            config.logger.warning(f"file already existed: {puls_file}")
                        puls_files[observed_quantity] = puls_file

                    if config.puls_type == 'pmode':
                        if type(config.prots) is list:
                            pulsation_grid_file = f"grid_summary/stormGrid_{grid}_rots.hdf"
                        else:
                            pulsation_grid_file = f"grid_summary/stormGrid_{grid}_rot{config.rotation_gyre}.hdf"
                        construct_pmode_pattern(config, grid, pulsation_grid_file, method, puls_file)
                        puls_files[observed_quantity] = puls_file

                        rotname = 'rotp'
                

                # Merge surface and pulsation info into one file
                output_name = f"{Path(puls_file).parent}/surface+{Path(puls_file).name}"
                if not Path(output_name).is_file():
                    sf.add_surface_to_puls_grid(puls_file, surface, output_name, grid_parameters=config.grid_parameters, rot_column=rotname)
                else:
                    config.logger.warning(f"file already existed: {output_name}")

            elif config.combined_fit is True:
                puls_files = {}
                for observed_quantity in observed_quantities:
                    if observed_quantity == 'frequency':
                        if type(config.prots) is list:
                            pulsation_grid_file = f"grid_summary/stormGrid_{grid}_rots.hdf"
                        else:
                            pulsation_grid_file = f"grid_summary/stormGrid_{grid}_rot{config.rotation_gyre}.hdf"
                        puls_file = f"extracted_freqs/{observed_quantity}_{config.star}_{grid}_{method}.hdf"
                        # the g-mode pattern (built first, observed_quantities lists 'period'
                        # before 'frequency') holds the per-model rotation for pmode_rotation='gmode'
                        construct_pmode_pattern(
                            config, grid, pulsation_grid_file, method, puls_file,
                            gmode_puls_file=puls_files.get("period"),
                        )
                        puls_files[observed_quantity] = puls_file

                    else:
                        pulsation_grid_files = {
                            mval: f"grid_summary/pulsationGrid_{grid}_rot{config.rotation_gyre}_k{config.kval}m{mval}.hdf"
                            for mval in config.mvals
                        }
                        asymp_objects = asymptotic_objects_per_m(config)
                        puls_file = f"extracted_freqs/{observed_quantity}_{config.star}_{grid}_{method}.hdf"
                        if not Path(puls_file).is_file():
                            bopp.construct_theoretical_puls_pattern_joint_m(
                                pulsation_grid_files, config.observations, method,
                                pattern_starting_pulsation=config.pattern_starting_pulsation[observed_quantity],
                                output_file=puls_file, asymp_objects=asymp_objects,
                                estimated_rotation=config.rotation_gyre, grid_parameters=config.grid_parameters,
                                nr_cpu=config.nr_cpu,
                            )
                        else:
                            config.logger.warning(f"file already existed: {puls_file}")
                        puls_files[observed_quantity] = puls_file

                # Merge surface + period puls-pattern + frequency puls-pattern into ONE file
                output_name = f"{Path(puls_file).parent}/surface+combined_{config.star}_{grid}_{method}.hdf"
                if not Path(output_name).is_file():
                    sf.add_surface_and_puls_to_grid(
                        puls_files, surface, output_name, grid_parameters=config.grid_parameters
                    )
                else:
                    config.logger.warning(f"file already existed: {output_name}")


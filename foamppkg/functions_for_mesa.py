"""A few helpful functions to process MESA output."""

import glob
import multiprocessing
import re
from functools import partial
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from . import support_functions as sf


################################################################################
def read_mesa_file(file_path, index_col=None):
    """
    Read in a mesa profile or history file and return 2 dictionaries.
    The first with the header info, and the second with the data.
    If the format is hdf5, assumes the attributes are the MESA header.
    If the format is not an hdf5 file, assumes the default MESA output format.
    This format is an ascii file delimited by whitespace with the following structure:
    line 1: header names
    line 2: header data
    line 3: blank
    line 4: main data names
    line >4:main data values

    Parameters
    ----------
    file_path: String
        The path to the MESA profile or history file to read in.
    index_col: Hashable
        Column to use as row labels.

    Returns
    ----------
    header: dict
        A dictionary holding the header info of the MESA file.
    data: dict
        A dictionary holding the data columns of the MESA file as numpy arrays.
    """
    if h5py.is_hdf5(file_path):
        return sf.read_hdf5(file_path)

    else:  # assumes the default MESA output format
        header_df = pd.read_table(file_path, sep="\s+", nrows=1, header=1)
        data_df = pd.read_table(file_path, sep="\s+", skiprows=3, header=1, index_col=index_col)

        header = {}
        for k in header_df.keys():
            header.update({k: header_df[k].to_numpy()[0]})
        data = {}
        for k in data_df.keys():
            data.update({k: data_df[k].to_numpy()})

        return header, data


def read_mesa_profile_surface(mesa_profile):
    """Read only the header and the first data row (zone 1 = surface) of a MESA ASCII profile."""
    with open(mesa_profile) as f:
        lines = [line for _, line in zip(range(7), f)]
    if len(lines) < 7:
        raise ValueError(f"only {len(lines)} lines, expected header and at least 1 data row (incomplete profile?)")
    header = dict(zip(lines[1].split(), lines[2].split()))
    surface = dict(zip(lines[5].split(), lines[6].split()))
    return header, surface


################################################################################
def calculate_number_densities(hist_file):
    """
    Calculate surface number densities for all isotopes in the MESA grid.
    All isotopes in the used nuclear network need to be be written in the history file,
    otherwise this will function give wrong numbers.

    Parameters
    ----------
    hist_file: string
        The path to the MESA history file.

    Returns
    ----------
    number_densities: dict
        Column keys specify the element (surf_X_per_N_tot), values are number densities of that element.
    """
    _, data = read_mesa_file(hist_file)
    element_list = {}
    number_densities = {}
    inverse_average_atomic_mass = np.zeros(len(data[list(data.keys())[0]]))
    for column_name in data.keys():
        if "_per_Mass_tot" in column_name:
            element_list.update({column_name: data[column_name]})
            inverse_average_atomic_mass += data[column_name]

    average_atomic_mass = inverse_average_atomic_mass ** (-1)
    for key in element_list.keys():
        number_densities.update({key.replace("_per_Mass_tot", "_per_N_tot"): element_list[key] * average_atomic_mass})

    return number_densities


################################################################################
def extract_surface_grid(
    mesa_profiles,
    output_file="surfaceGrid.hdf",
    parameters=["Z", "M", "logD", "aov", "fov", "Xc"],
    nr_cpu=None,
    additional_observables=None,
):
    """
    Extract 'logTeff', 'logL', 'logg', 'age', and extra requested info for each globbed MESA profile and write them to 1 large file.

    Parameters
    ----------
    mesa_profiles: string
        String to glob to find all the relevant MESA profiles.
    output_file: string
        Name (or path) for the file containing all the pulsation frequencies of the grid.
    parameters: list of strings
        List of parameters varied in the computed grid, so these are taken from the
        name of the profile files, and included in the output file containing the info of the whole grid.
    nr_cpu: int
        Number of worker processes to use in multiprocessing. The default 'None' will use the number returned by os.cpu_count().
    additional_observables: list of strings
        List of observables to add to the surface grid. Must correspond to mesa profile header-item names.
    """
    # Make list of extra observables requested by the user
    if additional_observables is None:
        additional_observables = []
    extras_to_be_extracted = [x for x in additional_observables if x not in ["logTeff", "logL", "logg", "age"]]

    extract_func = partial(_info_from_profiles_or_error, parameters=parameters, extra_header_items=extras_to_be_extracted)
    # Glob all the files, then iteratively send them to a pool of processors
    profiles = sorted(glob.glob(mesa_profiles))
    with multiprocessing.Pool(nr_cpu) as p:
        surface = p.imap(extract_func, profiles, chunksize=200)

        # Generate the directory for the output file and write the file afterwards
        Path(Path(output_file).parent).mkdir(parents=True, exist_ok=True)
        # make a new list, so 'parameters' is not extended before passing it on to 'info_from_profiles'
        header_parameters = list(parameters)
        header_parameters.extend(["logTeff", "logL", "logg", "age"])
        # Add the extra observables requested by the user
        header_parameters.extend(extras_to_be_extracted)

        # Make list of lists, put it in a dataframe, and write to a file.
        # Profiles that could not be read are skipped and listed in a separate file.
        data = []
        broken_profiles = []
        for profile, (line, error) in zip(profiles, surface):
            if error is None:
                data.append(line)
            else:
                broken_profiles.append(f"{profile}\t{error}")

    df = pd.DataFrame(data=data, columns=header_parameters)
    df.to_hdf(path_or_buf=output_file, key="surface_grid", format="table", mode="w")

    if broken_profiles:
        broken_file = broken_profiles_file(output_file)
        broken_file.write_text("\n".join(broken_profiles) + "\n")
        print(f"Skipped {len(broken_profiles)} of {len(profiles)} profiles that could not be read, listed in {broken_file}")


################################################################################
def info_from_profiles(mesa_profile, parameters, extra_header_items):
    """
    Extract 'logTeff', 'logL', 'logg', 'age', and extra requested info from a MESA profile and the model parameters from its filename.

    Parameters
    ----------
    mesa_profile: string
        path to the MESA profile
    parameters: list of strings
        List of input parameters varied in the computed grid, so these are read from the filename and included in returned line.
    extra_header_items: list of strings
        List of extra observables to add to the surface grid. Must correspond to mesa profile header-item names.

    Returns
    ----------
    line: string
        Line containing all the model- and surface parameters of the MESA profile.
    """
    param_dict = sf.get_param_from_filename(mesa_profile, parameters, values_as_float=True)

    if h5py.is_hdf5(mesa_profile):
        prof_header, prof_data = read_mesa_file(mesa_profile)
        prof_surface = {k: v[0] for k, v in prof_data.items()}
    else:
        prof_header, prof_surface = read_mesa_profile_surface(mesa_profile)

    logL = np.log10(float(prof_header["photosphere_L"]))
    logTeff = np.log10(float(prof_header["Teff"]))
    logg = float(prof_surface["log_g"])
    age = int(float(prof_header["star_age"]))

    line = []
    for p in parameters:
        line.append(param_dict[p])
    line.extend([logTeff, logL, logg, age])

    for obs in extra_header_items:  # Add the extra observables requested by the user
        item = float(prof_header[obs])
        line.append(item)

    return line


def broken_profiles_file(surface_grid_file):
    """Path of the file listing the profiles that 'extract_surface_grid' could not read."""
    return Path(surface_grid_file).with_name(f"{Path(surface_grid_file).stem}_broken_profiles.txt")


def read_broken_models(surface_grid_file):
    """
    Read the model names (profile filename without '.data', e.g. 'M1.98_D1_Z0.013_fov0.0225_Xc0.50968_')
    of the broken profiles listed next to the surface grid. Returns an empty set if there is no such file.
    """
    broken_file = broken_profiles_file(surface_grid_file)
    if not broken_file.is_file():
        return set()
    models = set()
    for line in broken_file.read_text().splitlines():
        if line.strip():
            models.add(Path(line.split("\t")[0]).name.removesuffix(".data"))
    return models


def exclude_models(files, models):
    """
    Remove the files belonging to any of the given model names, e.g. the GYRE output
    'rot0.1_M1.98_D1_Z0.013_fov0.0225_Xc0.50968_.data-freqs.HDF' of model 'M1.98_D1_Z0.013_fov0.0225_Xc0.50968_'.
    """
    if not models:
        return list(files)
    pattern = re.compile(rf"(?:^|_)(?:{'|'.join(re.escape(m) for m in models)})\.")
    kept, removed = [], []
    for f in files:
        (removed if pattern.search(Path(f).name) else kept).append(f)
    if removed:
        print(f"Excluded {len(removed)} files of broken profiles: {[Path(f).name for f in removed]}")
    return kept


def _info_from_profiles_or_error(mesa_profile, parameters, extra_header_items):
    """
    Wrapper around 'info_from_profiles' that does not raise on broken (e.g. incomplete) profiles.

    Returns
    ----------
    (line, error): tuple
        'line' as returned by 'info_from_profiles' and error None, or line None and a string
        describing the error if the profile could not be read.
    """
    try:
        return info_from_profiles(mesa_profile, parameters, extra_header_items), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"

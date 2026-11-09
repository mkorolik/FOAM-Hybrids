"""Helpful functions in general. Reading HDF5, processing strings, manipulating dataFrames..."""

import logging
import re
from pathlib import Path

import h5py
import pandas as pd

logger = logging.getLogger("logger.sf")


################################################################################
def split_line(line, sep):
    """
    Splits a string in 2 parts.

    Parameters
    ----------
    line: string
        String to split in 2.
    sep: string
        Separator where the string has to be split around.

    Returns
    ----------
    head: string
        Part 1 of the string before the separator.
    tail: string
        Part 2 of the string after the separator.
    """
    head, sep_, tail = line.partition(sep)
    assert sep_ == sep
    return head, tail


################################################################################
def substring(line, sep_first, sep_second):
    """
    Get part of a string between 2 specified separators.
    If second separator is not found, return everything after first separator.

    Parameters
    ----------
    line: string
        String to get substring from.
    sep_first: string
        First separator after which the returned substring should start.
    sep_first: string
        Second separator at which the returned substring should end.

    Returns
    ----------
    head: string
        Part of the string between the 2 separators.
    """
    head, tail = split_line(line, sep=sep_first)
    if sep_second not in tail:
        return tail
    head, tail = split_line(tail, sep=sep_second)
    return head


################################################################################
def get_param_from_filename(file_path, parameters, values_as_float=False):
    """
    Get parameters from filename

    Parameters
    ----------
    file_path : string
        Full path to the file
    parameters: list of string
        Names of parameters to extract from filename

    Returns
    ----------
    param_dict: dict
        Keys are strings describing the parameter, values are strings giving corresponding parameter values
    """

    param_dict = {}
    for parameter in parameters:
        try:
            p = substring(Path(file_path).stem, parameter, "_")
            if values_as_float:
                p = float(p)
            param_dict[parameter] = p
        except:
            logger.warning(
                f"In get_param_from_filename: parameter \"{parameter}\" not found in '{file_path}', value not added"
            )

    return param_dict


################################################################################
def read_hdf5(filename):
    """
    Read a HDF5-format file (e.g. GYRE)

    Parameters
    ----------
    filename : string
        Input file

    Returns
    ----------
    attributes: dict
        Dictionary containing the attributes of the file.
    data: dict
        Dictionary containing the data from the file as numpy arrays.
    """
    # Open the file
    with h5py.File(filename, "r") as file:
        # Read attributes
        attributes = dict(zip(file.attrs.keys(), file.attrs.values()))
        # Read datasets
        data = {}
        for k in file.keys():
            if isinstance(file[k], h5py.Dataset):
                data[k] = file[k][...]
    return attributes, data


################################################################################
def sign(x):
    """
    Returns the sign of a number as a string

    Parameters
    ----------
    x: float or int

    Returns
    ----------
    s: string
        A string representing the sign of the number
    """
    if abs(x) == x:
        return "+"
    else:
        return "-"


################################################################################
def get_subgrid_dataframe(file_to_read, fixed_params=None):
    """
    Read a hdf5 file containing the grid information as a pandas dataframe.
    Parameters can be fixed to certain values to filter out entries with other values of that parameter.

    Parameters
    ----------
    file_to_read: string
        path to the file to read
    fixed_params: dict
        keys are parameters to fix to the value specified in the dictionary

    Returns
    ----------
    df: pandas DataFrame
    """
    df = pd.read_hdf(file_to_read)

    if fixed_params is not None:
        for param in fixed_params.keys():
            indices_to_drop = df[df[param] != fixed_params[param]].index
            df.drop(indices_to_drop, inplace=True)
        df.reset_index(drop=True, inplace=True)

    return df


################################################################################
def add_surface_to_puls_grid(
    grid_frequencies,
    grid_surface,
    output_name=" grid_surface+freq.hdf",
    grid_parameters=["Z", "M", "logD", "aov", "fov", "Xc"],
    rot_column="rot"
):
    """
    Combine the output files with the frequencies and surface info of the grid in one new file,
    only keeping models that have entries in both the grid files.

    Parameters
    ----------
    grid_frequencies: string
        Path to the file containing the model input parameters and corresponding frequencies of the model.
    grid_surface: string
        Path to the file containing the model input parameters and corresponding surface info of the model.
    output_name: string
        Name of the generated file containing the combined info.
    grid_parameters: list of string
        List of the model parameters to use for matching the entries in the freq/surface file.
    """
    freq_df = pd.read_hdf(grid_frequencies)
    surface_df = pd.read_hdf(grid_surface)
    # Merge with surface info first, freq info second. Only keeping rows that both dataFrames have in common based on the 'on' columns.
    df_merged = pd.merge(surface_df, freq_df, how="inner", on=grid_parameters)

    _ = df_merged.pop("age")  # Don't add the age in the combined file

    # take the column with rotation and place it as the first column, and its error as second column
    col = df_merged.pop(rot_column)
    df_merged.insert(0, col.name, col)
    err_column = f"{rot_column}_err"
    if err_column in df_merged.columns:
        col = df_merged.pop(err_column)
        df_merged.insert(1, col.name, col)
    # write the merged dataFrame to a new file
    df_merged.to_hdf(path_or_buf=output_name, key="pulsation_surface_grid", format="table", mode="w")


def add_surface_and_puls_to_grid(
    puls_files,
    grid_surface,
    output_name="grid_surface+puls.hdf",
    grid_parameters=["Z", "M", "logD", "aov", "fov", "Xc"],
):
    """
    Merge the surface-info file with multiple pulsation-pattern files (e.g. one for
    g-mode periods, one for p-mode frequencies) into a single grid file, keeping only
    models present in all of them.

    Parameters
    ----------
    puls_files: dict
        Maps observable label ('period', 'frequency', ...) to the path of its puls_file.
    grid_surface: string
        Path to the file with surface info (Teff, logg, logL, ...).
    output_name: string
    grid_parameters: list of string
    rot_source: string
        Which entry of puls_files holds the physically meaningful 'rot'/'rot_err'
        (i.e. the one computed with TAR rescaling, typically 'period'). The rot/rot_err
        columns from all other puls_files are dropped before merging, since for those
        (asymptotic_object=None) they're just a constant placeholder, not a fit.
    """
    surface_df = pd.read_hdf(grid_surface)
    df_merged = surface_df

    for label, path in puls_files.items():
        puls_df = pd.read_hdf(path)
        df_merged = pd.merge(df_merged, puls_df, how="inner", on=grid_parameters)

    if "age" in df_merged.columns:
        _ = df_merged.pop("age")

    # keep rot/rot_err as the first two columns, same convention as add_surface_to_puls_grid
    # col = df_merged.pop("rot")
    # df_merged.insert(0, col.name, col)
    # col = df_merged.pop("rot_err")
    # df_merged.insert(1, col.name, col)
    for name in ["rotp", "rot", "rot_err"]:
        if name in df_merged.columns:
            col = df_merged.pop(name)
            df_merged.insert(0, col.name, col)


    df_merged.to_hdf(path_or_buf=output_name, key="pulsation_surface_grid", format="table", mode="w")
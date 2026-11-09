""" Functions to perform different kinds of maximum likelihood estimation for the models in a grid, and make correlation plots.
Note: The file with observations needs to hold temperature as Teff, although the analysis is done using the logTeff values."""

import logging
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import build_optimised_pattern as bop
from . import support_functions as sf

# Make a child logger of "logger" made in the top level script
logger = logging.getLogger("logger.mle_estimator")


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def calculate_likelihood(
    theory_file,
    observables=None,
    merit_function=None,
    obs_path=None,
    star_name=None,
    fixed_params=None,
    grid_parameters=None,
    max_components=None,
    sigma_theo_gmode=None,
    gmode_covariance="grid",
):
    """
    Perform a maximum likelihood estimation using the provided type of merit function on the list of observables.
    Writes a data file with the values of the merit function and input parameters of each model.
    Can select and continue the analysis of nested grids through the keyword 'fixed_params'.

    Parameters
    ----------
    obs_path: string
        Path to the tsv file with observations, with a column for each observable and each set of errors.
        Column names specify the observable, and "_err" suffix denotes that it's the error.
    theory_file: string
        Path to the hdf5 file with the theoretical model input parameters (first set of columns), frequency or period values (last set of columns),
        and possibly extra columns with additional observables (these columns should be in between the input parameters and frequency columns).
    observables: list of strings
        Which observables are included in the merit function.
        Must contain either 'f' (frequency), 'P' (period), or 'dP' (period-spacing) which will be computed for the period pattern.
        Can contain any additional observables that are added as columns in both the file with observations and the file with theoretical models.
    merit_function: string
        The type of merit function to use. Currently supports "CS and "MD" ("chi-squared" and "mahalanobis distance").
    star_name: string
        Name of the star, used in file naming.
    fixed_params: dict
        Only select and analyse the part of the theoretical grid with the specified parameter values.
        The keys specify for which parameters only the specified value should be selected.
    grid_parameters: list of string
        List of the parameters in the theoretical grid.
    sigma_theo_gmode: float or None
        Theoretical uncertainty on the g-mode periods or period spacings, in days, added
        to the diagonal of the variance-covariance matrix of the Mahalanobis distance
        (see merit_mahalanobis, sigma_theo_added). Only applied to period observables,
        and not to any additional observables. None keeps the original FOAM behaviour.
    gmode_covariance: string
        "grid" (default) adds sigma_theo_gmode to the grid-spread variance-covariance matrix,
        "diagonal" uses sigma_theo_gmode in its place (see gmode_md_covariance_kwargs).
    """
    if "f" in observables:
        observed_quantity = "frequency"
    elif "P" or "dP" in observables:
        observed_quantity = "period"
    # Read in the observed data and make an array of the observed observables
    obs_dataframe = pd.read_table(obs_path, sep="\s+", header=0, index_col="index")
    obs, obs_err, file_suffix_observables = create_obs_observables_array(obs_dataframe, observables)

    # set the name of the output file
    _, tail = sf.split_line(Path(theory_file).stem, star_name)
    filename = f"{star_name}{tail}_{merit_function}_{file_suffix_observables}"

    # Theoretical grid data
    theory_dataframe = sf.get_subgrid_dataframe(theory_file, fixed_params)
    
    # get the interruptions in the pattern, absolute index in dataframe
    missing_absolute = np.where(theory_dataframe.columns.to_series().str.contains(f"{observed_quantity}_missing"))[0]
    # get the interruptions in the pattern, index relative within pulsations
    missing_relative = np.where(
        theory_dataframe.filter(like=f"{observed_quantity}")
        .columns.to_series()
        .str.contains(f"{observed_quantity}_missing")
    )[0]
    # Remove columns of missing frequencies
    theory_dataframe = theory_dataframe.drop(columns=theory_dataframe.columns[missing_absolute])
    # Adjust indices for removed lines of missing frequencies
    missing_indices = [missing_relative[i] - i for i in range(len(missing_relative))]

    if observed_quantity == 'period':
        thetas = np.asarray(theory_dataframe.filter(["rot"] + ["rot_err"] + grid_parameters))
    elif observed_quantity == 'frequency':
        thetas = np.asarray(theory_dataframe.filter(["rotp"] + grid_parameters))
    theory_puls = np.asarray(theory_dataframe.filter(like=f"{observed_quantity}"))

    # Keep only the theoretical models without entries with value -1
    # (ignore models where the first or last freq is -1)
    keep = (theory_puls[:, 0] != -1) & (theory_puls[:, -1] != -1)
    new_theory = create_theory_observables_block(theory_dataframe, observables, missing_indices)[keep]
    new_thetas = thetas[keep]
    neg_value = np.unique(np.where(theory_puls == -1)[0])
    logger.debug(f"File: {theory_file}")
    logger.debug(f"observables      : {observables}")
    logger.debug(f"ignored -1 freqs : {len(neg_value)}")
    logger.debug(f"original #models : {len(theory_puls)}")
    logger.debug(f"remaining #models: {len(new_theory)}")
    if len(neg_value) > 0:
        logger.warning(
            f"""{len(neg_value)} models were discarded due to mismatches during the selection of theoretical frequencies.
                        This is likely due to the frequency range used for the theoretical calculations being too narrow."""
        )
    # Dictionary containing different merit functions
    switcher = {"CS": merit_chi2, "MD": merit_mahalanobis}

    # get the desired function from the dictionary. Returns the lambda function if option is not in the dictionary.
    selected_merit_function = switcher.get(
        merit_function,
        lambda x, y, z: sys.exit(logger.error("invalid type of maximum likelihood estimator")),
    )
    # Only the Mahalanobis distance inverts a variance-covariance matrix, so the rank cap
    # is meaningless for (and not accepted by) the chi-squared merit function.
    extra_kwargs = {"max_components": max_components} if selected_merit_function is merit_mahalanobis else {}
    if selected_merit_function is merit_mahalanobis and observed_quantity == "period":
        extra_kwargs.update(gmode_md_covariance_kwargs(obs_dataframe, observables, sigma_theo_gmode, gmode_covariance))
    merit_values = selected_merit_function(
        obs, obs_err, new_theory, fig_title=f"{filename}", star_name=star_name, **extra_kwargs
    )

    # Combine values and save the results
    # add an additional column for MLE 'meritValues'
    combined_data = np.concatenate((np.matrix(merit_values).T, new_thetas), axis=1)
    # put the data in a pandas DataFrame

    if observed_quantity == 'period':
        df = pd.DataFrame(data=combined_data, columns=["meritValue"] + ["rot"] + ["rot_err"] + grid_parameters)
        df = pd.merge(
            df,
            theory_dataframe.drop(list(theory_dataframe.filter(regex=f"{observed_quantity}").columns), axis=1),
            how="inner",
            on=["rot", "rot_err"] + grid_parameters,
        )
    else:
        df = pd.DataFrame(data=combined_data, columns=["meritValue"] + ["rotp"] + grid_parameters)
        df = pd.merge(
            df,
            theory_dataframe.drop(list(theory_dataframe.filter(regex=f"{observed_quantity}").columns), axis=1),
            how="inner",
            on=["rotp"] + grid_parameters,
        )
    df.to_hdf(path_or_buf=f"{os.getcwd()}/meritvalues/{filename}.hdf", key="merit_values", format="table", mode="w")


COMBINED_MERIT_FUNCTIONS = ("MD", "CS")


def combined_suffix(merit_function):
    """
    Suffix that tags the output files of a combined g+p fit with its merit function.

    The Mahalanobis distance keeps the plain "combined" suffix it had before the
    chi-squared option existed, so earlier runs and the scripts reading them keep
    working. The chi-squared files get "CS_combined".
    """
    if merit_function not in COMBINED_MERIT_FUNCTIONS:
        raise ValueError(
            f"combined_fit supports the merit functions {COMBINED_MERIT_FUNCTIONS}, got '{merit_function}'"
        )
    return "combined" if merit_function == "MD" else f"{merit_function}_combined"


def combined_observables(observable_seismic):
    """
    Split observable_seismic into the g-mode and p-mode observables of a combined fit.

    In a combined fit the g-modes are always the period pattern and the p-modes the
    frequencies, so each entry is assigned by which of the two it describes.
    """
    observables_gmode = [obs for obs in observable_seismic if obs in ("P", "dP")]
    observables_pmode = [obs for obs in observable_seismic if obs == "f"]
    if len(observables_gmode) != 1 or len(observables_pmode) != 1:
        raise ValueError(
            f"combined_fit needs exactly one of 'P'/'dP' for the g-modes and 'f' for the "
            f"p-modes in observable_seismic, got {observable_seismic}"
        )
    return observables_gmode, observables_pmode


def n_observables_combined(obs_path_gmode, obs_path_pmode, observables_gmode, observables_pmode):
    """
    Number of observables entering a combined g+p merit function, counted from the
    observation files: the g-mode periods or period spacings plus the p-mode frequencies.
    """
    obs_df_g = pd.read_table(obs_path_gmode, sep=r"\s+", header=0, index_col="index")
    obs_df_p = pd.read_table(obs_path_pmode, sep=r"\s+", header=0, index_col="index")
    n_g = len(create_obs_observables_array(obs_df_g, observables_gmode)[0])
    n_p = len(create_obs_observables_array(obs_df_p, observables_pmode)[0])
    return n_g + n_p


def calculate_likelihood_combined(
    theory_file,
    observables_gmode=['P'],
    observables_pmode=['f'],
    obs_path_gmode=None,
    obs_path_pmode=None,
    combine_method="sum",
    star_name=None,
    fixed_params=None,
    grid_parameters=None,
    sigma_theo_pmode=None,
    max_components=None,
    sigma_theo_gmode=None,
    merit_function="MD",
    gmode_covariance="grid",
):
    """Combine independently-computed g-mode and p-mode merit values into one merit
    value per model.

    merit_function selects how each block is scored:
      "MD" -- Mahalanobis distance, with the grid-spread variance-covariance matrix
              plus the observational errors (and any theoretical uncertainties below).
              Output: <star>_<grid>_<method>_combined.hdf
      "CS" -- plain chi-squared using only the observational errors. No variance-
              covariance matrix is built and sigma_theo_gmode, sigma_theo_pmode and
              max_components are ignored. Output: <star>_<grid>_<method>_CS_combined.hdf
    The mode-density penalty of the unidentified p-modes, and the inflated errors on
    those modes, are applied to both, so the two differ only in the merit function.

    sigma_theo_gmode is the theoretical uncertainty on the g-mode periods or period
    spacings, in days, added to the grid-spread variance-covariance matrix of MD_g
    (see merit_mahalanobis, sigma_theo_added). None keeps the original FOAM MD_g.
    With gmode_covariance="diagonal" it replaces the grid spread instead, the same way
    sigma_theo_pmode does for MD_p (see gmode_md_covariance_kwargs).

    sigma_theo_pmode is the theoretical frequency uncertainty used to build the
    p-mode variance-covariance matrix, in place of the spread of the grid. Set it
    whenever the pattern file carries "logdensity" columns: the mode-density
    penalty added below is on a -2 ln L scale, and a grid-spread MD is not, so
    leaving it as None lets the penalty swamp MD_p instead of correcting it.
    Calibrate its value by injection-recovery on the grid."""

    obs_df_g = pd.read_table(obs_path_gmode, sep="\s+", header=0, index_col="index")
    obs_g, obs_err_g, suffix_g = create_obs_observables_array(obs_df_g, observables_gmode)

    obs_df_p = pd.read_table(obs_path_pmode, sep="\s+", header=0, index_col="index")

    id_confidence_inflation = 3.0
    obs_df_p_weighted = obs_df_p.copy()
    obs_df_p_weighted.loc[obs_df_p_weighted["degree"].isna(), "frequency_err"] *= id_confidence_inflation

    obs_p, obs_err_p, suffix_p = create_obs_observables_array(obs_df_p_weighted, observables_pmode)

    _, tail = sf.split_line(Path(theory_file).stem, star_name)
    filename = f"{star_name}{tail}_{combined_suffix(merit_function)}"

    theory_dataframe = sf.get_subgrid_dataframe(theory_file, fixed_params)

    missing_abs = np.where(theory_dataframe.columns.to_series().str.contains("period_missing"))[0]
    missing_rel = np.where(
        theory_dataframe.filter(like="period").columns.to_series().str.contains("period_missing")
    )[0]
    theory_dataframe = theory_dataframe.drop(columns=theory_dataframe.columns[missing_abs])
    missing_indices = [missing_rel[i] - i for i in range(len(missing_rel))]

    theta_columns = ["rot", "rot_err", "rotp"] + grid_parameters   
    thetas = np.asarray(theory_dataframe.filter(theta_columns))
    assert thetas.shape[1] == len(theta_columns), (
        f"expected {len(theta_columns)} theta columns {theta_columns}, "
        f"got {thetas.shape[1]} -- check theory_dataframe.columns for a naming mismatch"
    )

    puls_g = np.asarray(theory_dataframe.filter(like="period"))
    puls_p = np.asarray(theory_dataframe.filter(like="frequency"))

    n_g_ok = sum(1 for row in puls_g if row[0] != -1 and row[-1] != -1)
    n_p_ok = sum(1 for row in puls_p if row[0] != -1 and row[-1] != -1)
    # print(f"g-mode complete: {n_g_ok}/{len(puls_g)}, p-mode complete: {n_p_ok}/{len(puls_p)}")

    keep = (puls_g[:, 0] != -1) & (puls_g[:, -1] != -1) & (puls_p[:, 0] != -1) & (puls_p[:, -1] != -1)
    new_theory_g = create_theory_observables_block(theory_dataframe, observables_gmode, missing_indices)[keep]
    new_theory_p = create_theory_observables_block(theory_dataframe, observables_pmode, missing_indices)[keep]
    new_thetas = thetas[keep]
    kept_rows = theory_dataframe.index[keep]

    logger.debug(f"models surviving joint g+p filter: {len(new_thetas)} / {len(puls_g)}")

    if len(new_thetas) == 0:
        logger.error(
            f"{filename}: no models satisfy both the g-mode and p-mode completeness "
            f"conditions -- check whether the g-mode and p-mode pattern ranges/observations "
            f"actually overlap for this grid."
        )
        return  # nothing to write for this (grid, method) combination

    if merit_function == "MD":
        # max_components is applied to both blocks: it is a no-op wherever the number of
        # observables is already at or below the cap, which is the usual case for the
        # p-modes, so the g-mode block does not need a separate keyword.
        merit_g = merit_mahalanobis(
            obs_g, obs_err_g, new_theory_g, fig_title=f"{filename}_g", star_name=star_name,
            max_components=max_components,
            **gmode_md_covariance_kwargs(obs_df_g, observables_gmode, sigma_theo_gmode, gmode_covariance),
        )
        merit_p = merit_mahalanobis(
            obs_p, obs_err_p, new_theory_p, fig_title=f"{filename}_p", star_name=star_name,
            sigma_theo=sigma_theo_pmode, max_components=max_components,
        )
    else:
        ignored = [
            name for name, value in (
                ("sigma_theo_gmode", sigma_theo_gmode),
                ("sigma_theo_pmode", sigma_theo_pmode),
                ("max_components", max_components),
            ) if value is not None
        ]
        if ignored:
            logger.info(f"{filename}: chi-squared uses only the observational errors, ignoring {ignored}")
        merit_g = merit_chi2(obs_g, obs_err_g, new_theory_g)
        merit_p = merit_chi2(obs_p, obs_err_p, new_theory_p)

    # Charge every model the uniform 1/N prior over which theoretical mode each
    # unidentified observation could have been. Without it a nearest-neighbour match
    # rewards models purely for having a denser spectrum, since the expected residual
    # falls as 1/(2*rho): in this grid the p-mode density rises by a factor ~5.5 from
    # the ZAMS to the hook, which is enough on its own to drive the fit to the TAMS.
    # Only the density half of the marginalised likelihood is added here, because the
    # residual half is already carried by the p-mode merit value (MD_p or chi2_p).
    logdensity_columns = list(theory_dataframe.filter(regex=r"^logdensity\d+$").columns)
    if logdensity_columns:
        density_penalty = 2.0 * theory_dataframe.loc[kept_rows, logdensity_columns].to_numpy(dtype=float).sum(axis=1)
        logger.debug(
            f"{filename}: mode-density penalty over {len(logdensity_columns)} unidentified mode(s), "
            f"range {density_penalty.min():.2f} to {density_penalty.max():.2f}"
        )
    else:
        density_penalty = np.zeros(len(new_thetas))

    if combine_method == "product":
        merit_values = merit_g * merit_p + density_penalty
    else:
        merit_values = merit_g + merit_p + density_penalty

    # Only the Mahalanobis distance has a variance-covariance matrix whose determinant
    # enters the likelihood normalisation (pipe4 AICc, pipe5 error ellipse).
    if merit_function == "MD":
        v_matrix_file = Path(f"{os.getcwd()}/V_matrix/{star_name}_determinant_conditionNr.tsv")
        df_v = pd.read_table(v_matrix_file, sep=r"\s+", header=0)
        ln_det_g = float(df_v.loc[df_v["method"] == f"{filename}_g", "ln(det(V))"].iloc[0])
        ln_det_p = float(df_v.loc[df_v["method"] == f"{filename}_p", "ln(det(V))"].iloc[0])
        ln_det_combined = ln_det_g + ln_det_p
        cond_combined = -1  # not meaningful for the combined "matrix" -- see earlier note
        n_components_g = df_v.loc[df_v["method"] == f"{filename}_g", "n_components"].iloc[0]
        n_components_p = df_v.loc[df_v["method"] == f"{filename}_p", "n_components"].iloc[0]

        with v_matrix_file.open("a") as f:
            f.write(f"{filename} \t {ln_det_combined:.2f} \t {cond_combined} \t {n_components_g}+{n_components_p} \n")

    combined_data = np.concatenate((np.matrix(merit_values).T, new_thetas), axis=1)
    df = pd.DataFrame(data=combined_data, columns=["meritValue"] + theta_columns)
    df = pd.merge(
        df,
        theory_dataframe.drop(list(theory_dataframe.filter(regex="period|frequency").columns), axis=1),
        how="inner", on=theta_columns,
    )
    df.to_hdf(path_or_buf=f"{os.getcwd()}/meritvalues/{filename}.hdf", key="merit_values", format="table", mode="w")



# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def create_theory_observables_array(theory_dataframe, index, observables_in, missing_indices):
    """
    Create an array of theoretical observables.

    Parameters
    ----------
    theory_dataframe: pandas dataFrame
        DataFrame containing the theoretical periods or frequencies (as the last columns), along with any additional
        columns containing extra observables.
    index: int
        Row index in the dataFrame of the theoretical model to make the array for.
    observables_in: list of strings
        Which observables are included in the returned array.
        Must contain either 'f' (frequency), 'P' (period), or 'dP' (period-spacing) which will be computed for the period pattern.
        Can contain any additional observables that are added as columns in both the file with observations and the file with theoretical models.
    missing_indices: list of int
        Contains the indices of the missing pulsations so that the period spacing pattern can be split around them.

    Returns
    ----------
    observables_out: numpy array, dtype=float
        The values of the specified observables for the model.
    """
    # Make a copy to leave the array handed to this function unaltered.
    observables = list(observables_in)
    if "P" in observables:
        # Cast to list before using asarray to prevent memory leak
        # add the periods to the output list
        observables_out = np.asarray(list(theory_dataframe.filter(like="period").loc[index]))
        observables.remove("P")

    elif "f" in observables:
        # Cast to list before using asarray to prevent memory leak
        # add the frequencies to the output list
        observables_out = np.asarray(list(theory_dataframe.filter(like="frequency").loc[index]))
        observables.remove("f")

    elif "dP" in observables:
        periods = np.asarray(theory_dataframe.filter(like="period").loc[index])
        observables_out = []
        for periods_part in np.split(periods, missing_indices):
            spacing, _ = bop.generate_spacing_series(periods_part)
            # switch back from seconds to days (so both P and dP are in days)
            observables_out = np.append(observables_out, np.asarray(spacing) / 86400)
        observables.remove("dP")

    # Add all other observables in the list from the dataFrame
    for observable in observables:
        observables_out = np.append(observables_out, np.asarray(theory_dataframe.loc[index, observable]))

    return observables_out


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def create_theory_observables_block(theory_dataframe, observables_in, missing_indices):
    """
    Create the arrays of theoretical observables for all models in the grid at once.
    Row i is the same as create_theory_observables_array(theory_dataframe, i, ...),
    but the columns are selected once for the whole grid instead of once per model.

    Parameters
    ----------
    theory_dataframe: pandas dataFrame
        DataFrame containing the theoretical periods or frequencies (as the last columns), along with any additional
        columns containing extra observables.
    observables_in: list of strings
        Which observables are included in the returned array.
        Must contain either 'f' (frequency), 'P' (period), or 'dP' (period-spacing) which will be computed for the period pattern.
        Can contain any additional observables that are added as columns in both the file with observations and the file with theoretical models.
    missing_indices: list of int
        Contains the indices of the missing pulsations so that the period spacing pattern can be split around them.

    Returns
    ----------
    observables_out: numpy ndarray (2D), dtype=float
        The values of the specified observables, one row per model.
    """
    # Make a copy to leave the array handed to this function unaltered.
    observables = list(observables_in)
    if "P" in observables:
        observables_out = theory_dataframe.filter(like="period").to_numpy(dtype=float)
        observables.remove("P")

    elif "f" in observables:
        observables_out = theory_dataframe.filter(like="frequency").to_numpy(dtype=float)
        observables.remove("f")

    elif "dP" in observables:
        periods = theory_dataframe.filter(like="period").to_numpy(dtype=float)
        # period spacings within each uninterrupted part of the pattern (in days, like P)
        observables_out = np.hstack(
            [np.abs(np.diff(periods_part, axis=1)) for periods_part in np.split(periods, missing_indices, axis=1)]
        )
        observables.remove("dP")

    # Add all other observables in the list from the dataFrame
    if observables:
        observables_out = np.hstack([observables_out, theory_dataframe[observables].to_numpy(dtype=float)])

    return observables_out


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def create_obs_observables_array(obs_dataframe, observables):
    """
    Create an array of the observed observables.

    Parameters
    ----------
    obs_dataframe: pandas dataFrame
        DataFrame containing the theoretical frequencies, periods, and any additional observables as columns, as well as columns with their errors.
        Column names specify the observable, and "_err" suffix denotes that it's the error.
    observables: list of strings
        Which observables are included in the returned array.
        Must contain either 'f' (frequency), 'P' (period), or 'dP' (period-spacing) which will be computed for the period pattern.
        Can contain any additional observables that are added as columns in both the file with observations and the file with theoretical models.

    Returns
    ----------
    observables_out: numpy array, dtype=float
        The values of the specified observables.
    observables_err_out: numpy array, dtype=float
        The errors on the values in observables_out.
    filename_suffix: string
        suffix for the filename, containing all the included observables, separated by '-'
    """
    # get the interruptions in the pattern
    missing_indices = np.where(obs_dataframe.index.isin(["f_missing"]))[0]
    # Adjust indices for removed lines of missing frequencies
    missing_indices = [missing_indices[i] - i for i in range(len(missing_indices))]
    # remove lines indicating missing frequencies (if they are present)
    if len(missing_indices) != 0:
        obs_dataframe = obs_dataframe.drop(index="f_missing")

    observables = list(observables)  # make a copy of the list, to not alter the one that was given to the function
    filename_suffix = ""

    if "P" in observables:
        observables_out = np.asarray(obs_dataframe["period"])
        observables_err_out = np.asarray(obs_dataframe["period_err"])
        filename_suffix = "P"
        observables.remove("P")

    elif "f" in observables:
        observables_out = np.asarray(obs_dataframe["frequency"])
        observables_err_out = np.asarray(obs_dataframe["frequency_err"])
        filename_suffix = "f"
        observables.remove("f")

    elif "dP" in observables:
        observables_out = []
        observables_err_out = []
        period = np.asarray(obs_dataframe["period"])
        period_err = np.asarray(obs_dataframe["period_err"])
        periods_parts = np.split(period, missing_indices)
        periods_err_parts = np.split(period_err, missing_indices)
        for periods, periods_err in zip(periods_parts, periods_err_parts):
            spacing, spacing_errs = bop.generate_spacing_series(periods, periods_err)
            # switch back from seconds to days (so both P and dP are in days)
            observables_out = np.append(observables_out, np.asarray(spacing) / 86400)
            observables_err_out = np.append(observables_err_out, np.asarray(spacing_errs) / 86400)

        filename_suffix = "dP"
        observables.remove("dP")

    if len(observables) > 0:
        filename_suffix += "+extra"
    # Add all other observables in the list from the dataFrame
    for observable in observables:
        logTeff = False
        if observable == "logTeff":
            # To read it as Teff from the observations datafile
            observable = "Teff"
            logTeff = True
        # since these columns have less entries, the dataFrame has NaNs in the empty rows
        obs = np.asarray(obs_dataframe[observable])
        # remove all entries that are NaN in the numpy array
        obs = obs[~np.isnan(obs)]
        obs_err = np.asarray(obs_dataframe[f"{observable}_err"])
        obs_err = obs_err[~np.isnan(obs_err)]

        # Convert observed Teff and error to log values
        if logTeff:
            logT = np.log10(obs)
            logT_err = obs_err / (obs * np.log(10))
            observables_out = np.append(observables_out, logT)
            observables_err_out = np.append(observables_err_out, logT_err)
        else:
            observables_out = np.append(observables_out, obs)
            observables_err_out = np.append(observables_err_out, obs_err)

    return observables_out, observables_err_out, filename_suffix


################################################################################
def gmode_sigma_theo_array(obs_dataframe, observables, sigma_theo_gmode):
    """
    Build the per-observable theoretical uncertainty for a g-mode observables array:
    sigma_theo_gmode on the periods or period spacings, and 0 on any additional
    observables (Teff, logL, ...) that follow them, since those are in other units.

    Parameters
    ----------
    obs_dataframe: pandas dataFrame
        The observations, as read for create_obs_observables_array.
    observables: list of strings
        The observables in the array, containing 'P' or 'dP'.
    sigma_theo_gmode: float or None
        Theoretical uncertainty on the periods or period spacings, in days.

    Returns
    ----------
    sigma: numpy array, dtype=float, or None
        None if sigma_theo_gmode is None, otherwise one value per observable.
    """
    if sigma_theo_gmode is None:
        return None
    seismic = [obs for obs in observables if obs in ("P", "dP")]
    n_seismic = len(create_obs_observables_array(obs_dataframe, seismic)[0])
    n_total = len(create_obs_observables_array(obs_dataframe, observables)[0])
    sigma = np.zeros(n_total)
    sigma[:n_seismic] = float(sigma_theo_gmode)
    return sigma


def gmode_md_covariance_kwargs(obs_dataframe, observables, sigma_theo_gmode, gmode_covariance="grid"):
    """
    Keyword arguments for merit_mahalanobis that set the variance-covariance matrix of the g-modes.

    gmode_covariance="grid" keeps the spread of the grid, with sigma_theo_gmode added to its
    diagonal: V = V_grid + sigma^2 I + V_obs (sigma_theo_added).
    gmode_covariance="diagonal" replaces the spread of the grid by sigma_theo_gmode:
    V = sigma^2 I + V_obs (sigma_theo), the same treatment sigma_theo_pmode gives the p-modes.
    In both cases sigma_theo_gmode only applies to the periods or period spacings; any
    additional observables (Teff, logL, ...) get no theoretical uncertainty.
    """
    sigma = gmode_sigma_theo_array(obs_dataframe, observables, sigma_theo_gmode)
    if gmode_covariance == "grid":
        return {"sigma_theo_added": sigma}
    if gmode_covariance == "diagonal":
        if sigma is None:
            raise ValueError("gmode_covariance='diagonal' needs a value for sigma_theo_gmode")
        return {"sigma_theo": sigma}
    raise ValueError(f"gmode_covariance should be 'grid' or 'diagonal', got {gmode_covariance!r}")


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def merit_chi2(Yobs, obs_err, YTheo, fig_title=None, star_name=None):
    """
    Calculate chi squared values for the given theoretical patterns

    Parameters
    ----------
    Yobs: numpy array, dtype=float
        observed values (period or frequency)
    obs_err: numpy array, dtype=float
        Errors on Yobs
    YTheo: numpy ndarray (2D), dtype=float
        Array of all theoretical patterns to calculate the chi squared value for.
    fig_title: None
        Should not be used in this function, but is to make it analogous to merit_mahalanobis()
        and enable the use of the lambda function.
    star_name: None
        Should not be used in this function, but is to make it analogous to merit_mahalanobis()
        and enable the use of the lambda function.
    Returns
    ----------
    chi2: numpy array, dtype=float
        chi squared values for the given theoretical values
    """
    chi2 = np.array([np.sum(((one_YTheo - Yobs) / obs_err) ** 2) for one_YTheo in YTheo])
    return chi2


################################################################################
def marginalised_unidentified_term(theory_dataframe, obs_dataframe, sigma_theo):
    """
    Calculate -2 ln L for the observed p-modes that have no identified degree,
    marginalising over which theoretical mode each one corresponds to instead of
    maximising over it.

    A nearest-neighbour match picks argmin_j |f_obs - f_j| and then charges only the
    residual, which hands a free advantage to any model with a denser spectrum. The
    marginalised alternative puts a uniform prior over the candidate modes,

        L_i = (1/N_i) * sum_j exp(-0.5 (f_i - f_j)^2 / sigma_theo^2)

    and, because sigma_theo is far below the spacing between candidates, the sum is
    dominated by the nearest one:

        -2 ln L_i ~= (f_i - f_nearest)^2 / sigma_theo^2  +  2 ln rho_i  +  constant

    with rho_i the local mode density stored by
    build_optimized_pattern_pmodes.match_unidentified_nearest_any_degree. The
    constant is the same for every model and drops out of the comparison; the
    2 ln rho_i term is the point, since it charges a model for the flexibility a
    denser spectrum buys it.

    Note that this returns a quantity on a -2 ln L scale, set by sigma_theo. A
    Mahalanobis distance is normalised by the variance-covariance matrix of the
    grid instead, so the two are not on a common scale and adding them together
    is not meaningful without deciding how to reconcile that.

    Parameters
    ----------
    theory_dataframe: pandas DataFrame
        Selected theoretical pattern, holding a "frequency{i}" and a matching
        "logdensity{i}" column for every unidentified observed mode i.
    obs_dataframe: pandas DataFrame
        The p-mode observations, indexed as in the observations file.
    sigma_theo: float
        The theoretical frequency uncertainty, i.e. how far a model frequency can
        systematically sit from reality. This is NOT the observational error bar:
        that is orders of magnitude below the model-to-model spread and would make
        every model in the grid infinitely improbable. Calibrate it by
        injection-recovery on the grid.

    Returns
    ----------
    total: numpy array, dtype=float
        The summed -2 ln L contribution of the unidentified modes, per model.
    """
    total = np.zeros(len(theory_dataframe))
    for column in theory_dataframe.filter(regex=r"^logdensity\d+$").columns:
        i = column.replace("logdensity", "")
        obs_freq = obs_dataframe.loc[obs_dataframe.index[int(i) - 1], "frequency"]
        residual = theory_dataframe[f"frequency{i}"].to_numpy(dtype=float) - obs_freq
        total += (residual / sigma_theo) ** 2 + 2.0 * theory_dataframe[column].to_numpy()
    return total


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def merit_mahalanobis(
    Yobs, obs_err, YTheo, generate_output=True, fig_title=None, star_name=None, sigma_theo=None,
    max_components=None, sigma_theo_added=None,
):
    """
    Calculate mahalanobis distance (MD) values for the given theoretical patterns.

    Parameters
    ----------
    Yobs: numpy array, dtype=float
        observed values (period or frequency)
    obs_err: numpy array, dtype=float
        Errors on Yobs
    YTheo: numpy array of arrays of floats
        Array of all theoretical patterns to calculate the MD value for.
    generate_output: boolean
        Flag to write output and plot the variance-covariance matrix
    fig_title: string
        The name of the figure to be created.
    star_name: string
        The name of the analysed star, for file naming purposes.
    sigma_theo: float, numpy array or None
        If None (the default), the variance-covariance matrix is built from the
        spread of the values across the grid, which is the standard FOAM
        behaviour. If a value is given, a diagonal matrix of that theoretical
        uncertainty is used instead (a float for the same value on every observable,
        or an array of length N for a value per observable).

        Use it when the grid spread does not represent a theoretical uncertainty.
        With only a handful of observables the spread across the grid measures how
        different the models in the grid are from one another, not how far any one
        of them can sit from the star, and it is typically far larger. That makes
        the resulting MD nearly flat between models, so it can no longer be added
        to terms that really are on a -2 ln L scale (a mode-density penalty, for
        instance) without the other term dominating it entirely.
    max_components: int or None
        Cap on the number of eigendirections of the variance-covariance matrix
        that are inverted, passed to regularized_inverse_and_logdet. Set it to
        the number of parameters generating the theoretical patterns when the
        observables outnumber them; see that function for why. Ignored when
        sigma_theo is given, since a diagonal matrix is already full rank.
    sigma_theo_added: float, numpy array or None
        Theoretical uncertainty added in quadrature to the diagonal of the
        variance-covariance matrix, on top of the grid spread (or of sigma_theo) and
        the observational errors: V = V_grid + sigma_theo_added^2 I + V_obs. Unlike
        sigma_theo it keeps the correlations of the grid spread, and only accounts
        for how far the models can sit from the star because of physics the grid does
        not vary. Without it, a residual of a few tens of seconds in the period
        spacings is divided by the smallest eigenvalues of V_grid, which are orders of
        magnitude smaller, and dominates the MD. Give a float for the same value on
        every observable, or an array of length N for a value per observable (0 for
        observables it should not apply to). Same units as Yobs. None (the default)
        leaves V as in the original FOAM.

    Returns
    ----------
    MD: numpy array, dtype=float
        Mahalanobis distances for the given theoretical patterns.
    """
    # Convert to matrix format (np.matrix is not recommended, use array and newaxis instead)
    YobsMat = np.array(Yobs)[np.newaxis].T
    YTheoMat = np.array(YTheo)[np.newaxis].T

    # number of grid points
    q = np.shape(YTheo)[0]
    # number of observed values
    N = len(Yobs)

    if sigma_theo is None:
        # Calculate the average on the theoretical values (e.g. frequencies)
        # over the entire grid. Returns a vector of dimension N x 1
        Yav = YTheoMat.mean(1)

        # Calculate the variance-covariance matrix
        v_matrix = np.zeros((N, N))
        for i in range(q):
            difference = np.subtract(YTheoMat[:, i], Yav)
            v_matrix += np.matmul(difference, difference.T)
        v_matrix = v_matrix / float(q - 1)
    else:
        # Uncorrelated theoretical uncertainty of the stated size, in place of the
        # grid spread. MD then reduces to a plain chi-squared in units of sigma_theo.
        v_matrix = np.diag(np.broadcast_to(np.asarray(sigma_theo, dtype=float), (N,)) ** 2.0)

    if sigma_theo_added is not None:
        # Model uncertainty the grid spread does not contain, added in quadrature.
        v_matrix = v_matrix + np.diag(np.broadcast_to(np.asarray(sigma_theo_added, dtype=float), (N,)) ** 2.0)

    # Include observational errors in the variance-covariance matrix
    v_matrix = v_matrix + np.diag(obs_err**2.0)
    # Regularize: invert only the eigendirections that carry real signal above the
    # numerical/interpolation noise floor of the grid, rather than the raw inverse
    # (which blows up on the near-zero eigenvalues a large, correlated observable set
    # produces, turning tiny grid noise into huge, physically meaningless distances).
    v_matrix_pinv, logdet, n_components = regularized_inverse_and_logdet(
        v_matrix, max_components=max_components
    )
    # check if positive definite and make figure
    check_matrix(v_matrix, logdet, n_components, generate_output=generate_output, fig_title=fig_title, star_name=star_name)
    # Calculate Mahalanobis distances
    MD = np.zeros(q)
    for i in range(q):
        diff = YTheoMat[:, i] - YobsMat
        MD[i] = np.matmul(np.matmul(diff.T, v_matrix_pinv), diff)[0][0]

    return MD


################################################################################
def regularized_inverse_and_logdet(v_matrix, rcond=1e-6, max_components=None):
    """
    Build a regularized pseudo-inverse and pseudo-log-determinant of a
    (possibly near-singular) covariance matrix by truncating its eigendecomposition
    to the eigenvalues that are at least `rcond` times the largest eigenvalue,
    and discarding the rest.

    A covariance matrix built from many correlated observables (e.g. individual
    periods/period-spacings from a grid with few free parameters) typically has a
    handful of large eigenvalues carrying the real signal, followed by a long tail
    of near-zero eigenvalues that are dominated by grid/interpolation noise rather
    than physical information. Inverting those noise eigenvalues directly amplifies
    that noise by up to 1/eigenvalue, swamping the Mahalanobis distance for every
    model with a similar noise floor instead of distinguishing good fits from bad
    ones. Restricting the inverse (and the determinant used in the AICc/likelihood
    normalisation) to the well-conditioned eigendirections avoids that.

    Parameters
    ----------
    v_matrix: numpy ndarray (2D), dtype=float
        Variance-covariance matrix.
    rcond: float
        Eigenvalues smaller than rcond * (largest eigenvalue) are treated as
        noise and dropped. Caps the effective condition number of the retained
        eigenspace at 1/rcond.
    max_components: int or None
        Hard cap on how many eigendirections are retained, applied on top of
        rcond. None (the default) leaves the truncation to rcond alone.

        A grid-spread covariance cannot carry more information than the number
        of parameters that generated the patterns: with p free parameters the
        theoretical patterns lie on a p-dimensional manifold in N-observable
        space, so at most p of the N eigenvalues describe real theoretical
        variation and the remaining N-p describe the curvature of that manifold
        and the discretisation of the grid. Those are typically several orders
        of magnitude smaller but still far above any rcond chosen to guard
        against numerical singularity alone, so they survive the rcond cut and
        are then inverted -- dividing a residual by a "sigma" that is an
        artefact of the grid rather than an uncertainty. Setting this to the
        number of generating parameters keeps the eigendirections that mean
        something and drops the rest. Capping at a value >= N is a no-op, so it
        is safe to apply the same cap to observable sets too small to be
        rank-deficient.

    Returns
    ----------
    v_matrix_pinv: numpy ndarray (2D), dtype=float
        Truncated pseudo-inverse of v_matrix.
    logdet: float
        Sum of the log of the retained eigenvalues (pseudo-log-determinant).
    n_components: int
        Number of eigenvalues retained.
    """
    eigvals, eigvecs = np.linalg.eigh(v_matrix)
    # eigh returns eigenvalues in ascending order; flip to descending.
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]

    keep = eigvals >= rcond * eigvals[0]
    n_components = max(int(np.sum(keep)), 1)
    if max_components is not None:
        n_components = max(min(n_components, int(max_components)), 1)
    kept_vals = eigvals[:n_components]
    kept_vecs = eigvecs[:, :n_components]

    v_matrix_pinv = kept_vecs @ np.diag(1.0 / kept_vals) @ kept_vecs.T
    logdet = float(np.sum(np.log(kept_vals)))
    return v_matrix_pinv, logdet, n_components


################################################################################
def check_matrix(v_matrix, logdet, n_components, generate_output=True, fig_title="Vmatrix", star_name=None):
    """
    Check that the eigenvalues of the variance-covariance matrix are all positive,
    since this means the matrix is positive definite. Write its (regularized,
    truncated) log-determinant, condition number, and number of retained
    eigencomponents to a tsv file. Create and save a figure of the variance-covariance matrix.

    Parameters
    ----------
    v_matrix: numpy ndarray (2D), dtype=float
        Variance-covariance matrix
    logdet: float
        Regularized pseudo-log-determinant from regularized_inverse_and_logdet,
        computed from only the retained (well-conditioned) eigenvalues.
    n_components: int
        Number of eigenvalues retained in logdet/the pseudo-inverse.
    output: boolean
        Flag to write output and plot the variance-covariance matrix
    fig_title: string
        The name of the figure to be created.
    star_name: string
        The name of the analysed star, for file naming purposes.
    """
    # If all eigenvalues are >0, it is positive definite
    if np.all(np.linalg.eigvals(v_matrix) > 0) == False:
        logger.error("V matrix is possibly not positive definite (since eigenvalues are not all > 0)")
        sys.exit(1)
    logger.debug(f"max(v_matrix) = {np.max(v_matrix)}")

    if generate_output is True:
        file_path = Path(f"{os.getcwd()}/V_matrix/{star_name}_determinant_conditionNr.tsv")
        file_path.parent.mkdir(parents=True, exist_ok=True)
        if not file_path.is_file():
            with file_path.open("w") as file:
                file.write(f"method \t ln(det(V)) \t condition_number \t n_components \n")
        with file_path.open("a") as file:
            file.write(
                f"{fig_title} \t {logdet:.2f} \t {np.linalg.cond(v_matrix):.2f} \t {n_components}/{v_matrix.shape[0]} \n "
            )

        # Do *10^4 to get rid of small values, and put this in the colorbar label
        im = plt.imshow(v_matrix * 10**4, aspect="auto", cmap="Reds")
        plt.ylabel(rf"obs {v_matrix.shape[0]}  $\leftarrow$ obs 1", size=14)
        plt.xlabel(rf"obs 1 $\rightarrow$ obs {v_matrix.shape[0]} ", size=14)

        cbar = plt.colorbar(im)
        cbar.ax.set_ylabel(r"[d$^{2} 10^{-4}$]", rotation=90, labelpad=15, size=14)
        cbar.ax.tick_params(labelsize=14)
        plt.title("Variance covariance matrix")
        plt.tick_params(labelsize=14)
        plt.tight_layout()
        plt.savefig(f"{os.getcwd()}/V_matrix/{fig_title}.png")
        plt.clf()
        plt.close("all")

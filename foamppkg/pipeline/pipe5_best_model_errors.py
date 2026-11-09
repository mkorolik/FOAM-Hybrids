"""Calculate the 2 sigma uncertainty region of the maximum likelihood solution using Bayes' theorem."""

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from foamppkg import maximum_likelihood_estimator as mle
from foamppkg import support_functions as sf
from foamppkg.pipeline.pipeline_config import config


def run(config):
    ################################################################################
    n_dict = config.n_dict  # number of observables
    sigma = 2
    percentile = {1: 0.68, 2: 0.95, 3: 0.997}

    if config.observable_additional is not None:
        extra_obs = "+extra"
    else:
        extra_obs = ""


    ################################################################################
    def likelihood_chi2(chi2):
        """Likelihood function of reduced chi-squared"""
        return np.exp(-0.5 * chi2 / (n_dict[obs] - config.k))


    def likelihood_md(md):
        """Likelihood function of the mahalanobis distance"""
        df_aicc_md = pd.read_table(f"V_matrix/{config.star}_determinant_conditionNr.tsv", sep="\s+", header=0)
        ln_det_v = float((df_aicc_md.loc[df_aicc_md["method"] == f"{config.star}_{analysis}", "ln(det(V))"]).iloc[0])
        return np.exp(-0.5 * (md + config.k * np.log(2 * np.pi) + ln_det_v))


    ################################################################################
    if config.n_sigma_box != None:
        directory_prefix = f"{config.n_sigma_box}sigmaBox_"
    else:
        directory_prefix = ""

    if config.combined_fit is False:
        for merit in config.merit_functions:
            for obs in config.observable_seismic:
                obs += extra_obs
                files = glob.glob(f"{directory_prefix}meritvalues/*{merit}_{obs}.hdf")
                for file in sorted(files):
                    Path_file = Path(file)
                    output_name = Path_file.with_stem(f"{Path_file.stem}_{sigma}sigma-error-ellipse")
                    # Don't duplicate if file is already present
                    if output_name.is_file():
                        config.logger.warning(f"file already existed: {output_name}")
                        continue

                    star_name, analysis = sf.split_line(Path_file.stem, "_")
                    df = pd.read_hdf(file)
                    df = df.sort_values("meritValue", ascending=True)

                    # Dictionary containing different likelihood functions
                    switcher = {"CS": likelihood_chi2, "MD": likelihood_md}
                    # get the desired function from the dictionary. Returns the lambda function if option is not in the dictionary.
                    likelihood_function = switcher.get(
                        merit, lambda x: sys.exit(config.logger.error(f"invalid type of maximum likelihood estimator:{merit}"))
                    )

                    probabilities = {}
                    for column_name in config.free_parameters:
                        probabilities.update({column_name: {}})
                        # construct dictionary
                        for value in df[column_name].unique():
                            probabilities[column_name].update({value: 0})
                        # sum over all occurrences of parameter values
                        for value in df[column_name]:
                            probabilities[column_name][value] += 1
                        # divide by total number of models to get probabilities
                        for value in df[column_name].unique():
                            probabilities[column_name][value] = probabilities[column_name][value] / len(df)

                    total_probability = 0
                    # calculate the denominator
                    for i in range(len(df)):
                        prob = likelihood_function(df.iloc[i]["meritValue"] - df.iloc[0]["meritValue"])
                        # prob = likelihood_function( df.iloc[i]['meritValue'] )

                        for column_name in config.free_parameters:
                            value = df.iloc[i][column_name]
                            prob = prob * probabilities[column_name][value]
                        total_probability += prob
                    p = 0
                    for i in range(len(df)):
                        prob = likelihood_function(df.iloc[i]["meritValue"] - df.iloc[0]["meritValue"])
                        # prob = likelihood_function( df.iloc[i]['meritValue'] )
                        for column_name in config.free_parameters:
                            value = df.iloc[i][column_name]
                            prob = prob * probabilities[column_name][value]

                        p += prob / total_probability
                        if p >= percentile[sigma]:
                            # Write all models enclosed within the error ellipse to a separate file
                            df.iloc[: i + 1].to_hdf(
                                path_or_buf=output_name, key="models_in_2sigma_error_ellipse", format="table", mode="w"
                            )
                            config.logger.debug(f"---------- {analysis} ---------- {i+1} --- {p}")
                            break


    elif config.combined_fit is True:
        # Build the expected file names instead of globbing the directory, which would
        # also pick up the error-ellipse files this pipe writes next to its inputs.
        for merit in config.merit_functions:
            suffix = mle.combined_suffix(merit)
            for grid in config.grids:
                for method in config.pattern_methods:
                    analysis = f"{grid}_{method}_{suffix}"
                    stem = f"{config.star}_{analysis}"
                    file = Path(f"{directory_prefix}meritvalues/{stem}.hdf")
                    output_name = file.with_stem(f"{stem}_{sigma}sigma-error-ellipse")
                    # Don't duplicate if file is already present
                    if output_name.is_file():
                        config.logger.warning(f"file already existed: {output_name}")
                        continue
                    if not file.is_file():
                        config.logger.error(f"merit values not found, skipping: {file}")
                        continue

                    df = pd.read_hdf(file)
                    df = df.sort_values("meritValue", ascending=True)

                    if merit == "MD":
                        df_v = pd.read_table(f"V_matrix/{config.star}_determinant_conditionNr.tsv", sep=r"\s+", header=0)
                        ln_det_v_rows = df_v.loc[df_v["method"] == stem, "ln(det(V))"]
                        if ln_det_v_rows.empty:
                            raise ValueError(
                                f"no ln(det(V)) entry for {stem} in V_matrix/{config.star}_determinant_conditionNr.tsv; "
                                f"rerun pipe2 so it matches the merit values"
                            )
                        ln_det_v = float(ln_det_v_rows.iloc[0])

                        def likelihood_function(md):
                            """Likelihood function of the mahalanobis distance"""
                            return np.exp(-0.5 * (md + config.k * np.log(2 * np.pi) + ln_det_v))

                    elif merit == "CS":
                        n_obs = mle.n_observables_combined(
                            config.observations, config.observations_pmode,
                            *mle.combined_observables(config.observable_seismic),
                        )

                        def likelihood_function(chi2):
                            """Likelihood function of reduced chi-squared, over all g-mode + p-mode observables"""
                            return np.exp(-0.5 * chi2 / (n_obs - config.k))

                    write_error_ellipse(
                        df, likelihood_function, config.free_parameters, percentile[sigma], output_name,
                        config.logger, analysis,
                    )


def write_error_ellipse(df, likelihood_function, free_parameters, percentile, output_name, logger, analysis):
    """
    Write the best models that together hold `percentile` of the posterior probability.

    Each model is weighted by its likelihood, relative to the best model, times the
    fraction of the grid sharing each of its free-parameter values. This is the same
    computation as the loops in the single-pulsation-type branch, done per column.

    Parameters
    ----------
    df: pandas DataFrame
        Merit values of the models, sorted with the best model first.
    likelihood_function: function
        Maps the merit value difference with the best model to a likelihood.
    free_parameters: list of strings
        Columns of the free parameters of the grid.
    percentile: float
        Probability that the written models enclose.
    output_name: Path
        File to write the enclosed models to.
    """
    prob = likelihood_function((df["meritValue"] - df["meritValue"].iloc[0]).to_numpy(dtype=float))
    for column_name in free_parameters:
        fraction = df[column_name].map(df[column_name].value_counts(normalize=True))
        prob = prob * fraction.to_numpy(dtype=float)

    cumulative = np.cumsum(prob) / np.sum(prob)
    enclosed = np.nonzero(cumulative >= percentile)[0]
    # Rounding can leave the last cumulative value a hair below 1, so fall back to all models.
    n_models = enclosed[0] + 1 if len(enclosed) > 0 else len(df)
    # Write all models enclosed within the error ellipse to a separate file
    df.iloc[:n_models].to_hdf(path_or_buf=output_name, key="models_in_2sigma_error_ellipse", format="table", mode="w")
    logger.debug(f"---------- {analysis} ---------- {n_models} --- {cumulative[n_models - 1]}")

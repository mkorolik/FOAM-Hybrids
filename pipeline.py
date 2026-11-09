""" Top level script to run the pipeline sequentially, copy this script to the folder where you want to run the analysis.
Comment specific imports if you don't want to repeat them on repeated runs."""

if __name__ == "__main__":
    import importlib
    import os
    from pathlib import Path

    from foamppkg.pipeline import pipeline_config
    from foamppkg.pipeline import (
            pipe2_calculate_likelihood,
            pipe3_add_constraints,
            pipe4_AICc,
            pipe5_best_model_errors,
            pipe6_corner_plots,
            pipe7_table_best_models,
        )    
    from foamppkg.foam_pmodes import pipe0, pipe1

    print('running config')
    pipeline_config.config = pipeline_config.PipelineConfig()
    

    # custom p-mode pipeline components
    print('Running Pipe 0')
    pipe0(pipeline_config.config)
    print('Running Pipe 1')
    pipe1(pipeline_config.config)
    print('Running Pipe 2')
    pipe2_calculate_likelihood.run(pipeline_config.config)
    print('Running Pipe 3')
    pipe3_add_constraints.run(pipeline_config.config)
    print('Running Pipe 4')
    pipe4_AICc.run(pipeline_config.config)
    print('Running Pipe 5')
    pipe5_best_model_errors.run(pipeline_config.config)
    print('Running Pipe 6')
    pipe6_corner_plots.run(pipeline_config.config)
    print('Running Pipe 7')
    pipe7_table_best_models.run(pipeline_config.config)

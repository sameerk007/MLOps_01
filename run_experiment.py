from azureml.pipeline.core import Pipeline, PipelineData ,PipelineParameter
from azureml.pipeline.steps import PythonScriptStep
from azureml.core import Workspace, Experiment, Environment , RunConfiguration


# Load your Azure ML workspace
ws = Workspace.from_config()

# Define an Azure ML environment with required dependencies
env = ws.environments['mlops_01_env_from_pip_local']

# Define output directory for processed data
processed_data = PipelineParameter(name="processed_data", default_value="./processed_data")


# Create A Run Configuration
aml_run_config = RunConfiguration()
aml_run_config.target = "MLOps-01"  #The compute_targets available 
aml_run_config.environment = ws.environments['mlops_01_env_from_pip_local']  #The enviornmet 



# Create the steps
load_data_step = PythonScriptStep(name="load_data_step",
                                  script_name="load_data.py",
                                  arguments=["--output_folder", processed_data],
                                  source_directory="data_ingestion",
                                  runconfig=aml_run_config,
                                 )

process_data_step = PythonScriptStep(name="process_data_step",
                                     script_name="process_data.py",
                                     arguments=["--input_folder", processed_data, "--output_folder", processed_data],
                                     source_directory="data_preprocessing",
                                     runconfig=aml_run_config,
                                    )

train_model_step = PythonScriptStep(name="train_model_step",
                                   script_name="train_model.py",
                                   arguments=["--input_folder", processed_data, "--output_folder", processed_data],
                                   source_directory="modeling",
                                   runconfig=aml_run_config,
                                  )



# Specify the dependencies between the steps
process_data_step.run_after(load_data_step)
train_model_step.run_after(process_data_step)

# Create the pipeline
steps = [load_data_step, process_data_step, train_model_step]
pipeline = Pipeline(workspace=ws, steps=steps)

# Create an experiment
experiment = Experiment(workspace=ws, name="iris-experiment")

# Submit the pipeline for execution
pipeline_run = experiment.submit(pipeline)
pipeline_run.wait_for_completion()

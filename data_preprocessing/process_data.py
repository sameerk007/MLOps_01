import pandas as pd
from sklearn.preprocessing import StandardScaler
from azureml.core import Dataset, Run

def main():
    # Load the registered dataset
    iris_dataset = Dataset.get_by_name(workspace=Run.get_context().experiment.workspace, name="iris_data")

    # Convert to Pandas DataFrame
    df = iris_dataset.to_pandas_dataframe()

    # # Perform data preprocessing (e.g., scaling)
    # scaler = StandardScaler()
    # scaled_data = scaler.fit_transform(df)
    scaled_data = df.copy()

    # Get the default datastore of the workspace
    datastore = Run.get_context().experiment.workspace.get_default_datastore()

    # Save the processed data to a new dataset (or a file)
    processed_dataset = Dataset.Tabular.register_pandas_dataframe(scaled_data,
                                                                target=datastore,
                                                                name="processed_iris_dataset",
                                                                )


if __name__ == "__main__":
    main()                                                            
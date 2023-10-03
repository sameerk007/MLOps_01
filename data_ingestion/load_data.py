# load_data.py
from azureml.core import Dataset, Run
from sklearn import datasets
import pandas as pd


def main():
    # Load the dataset (e.g., Iris dataset)
    iris = datasets.load_iris()
    
    # set the iris data into a pandas df
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target

    # Get the default datastore of the workspace
    datastore = Run.get_context().experiment.workspace.get_default_datastore()

    # Register the dataset in the Azure ML workspace
    iris_dataset = Dataset.Tabular.register_pandas_dataframe(dataframe=df,
                                                             target=datastore,
                                                             name='iris_data',
                                                            )

if __name__ == "__main__":
    main()
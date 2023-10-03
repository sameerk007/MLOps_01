from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from azureml.core import Run, Dataset
import os
import joblib


def main():
    # Load the registered processed dataset
    processed_dataset = Dataset.get_by_name(workspace=Run.get_context().experiment.workspace, name="processed_iris_dataset")

    # Convert to Pandas DataFrame
    df = processed_dataset.to_pandas_dataframe()

    # Separate features and target
    X = df.drop(columns=["target"])
    y = df["target"]


    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a simple Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = clf.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    # Save the trained model
    model_dir = "outputs"
    os.makedirs(model_dir, exist_ok=True)
    model_file = os.path.join(model_dir, "model.pkl")
    joblib.dump(clf, model_file)


if __name__ == "__main__":
    main()    
import argparse
import os
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def main():
    parser = argparse.ArgumentParser(description="Train a simple Random Forest Classifier.")
    parser.add_argument("--output_folder", type=str, help="Folder to save model and results.")
    args = parser.parse_args()

    # Load the Iris dataset
    iris = datasets.load_iris()
    X, y = iris.data, iris.target

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

    # Save the model to the specified output folder
    model_filename = os.path.join(args.output_folder, "model.pkl")
    joblib.dump(clf, model_filename)

if __name__ == "__main__":
    main()

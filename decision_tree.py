print("DECISION TREE FILE IS RUNNING")

import os
import sys
import joblib

from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# Find the src folder
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Find the main project folder
PROJECT_DIR = os.path.dirname(CURRENT_DIR)

# Allow importing preprocessing.py
sys.path.insert(0, CURRENT_DIR)

from preprocessing import prepare_data, create_preprocessor


# --------------------------------------------------
# FILE LOCATIONS
# --------------------------------------------------

DATA_PATH = os.path.join(
    PROJECT_DIR,
    "data",
    "dataset_heart.csv"
)

MODEL_DIR = os.path.join(
    PROJECT_DIR,
    "models"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "decision_tree.pkl"
)


# --------------------------------------------------
# TRAIN DECISION TREE
# --------------------------------------------------

def train_decision_tree():

    print("Starting Decision Tree training...")

    # Check dataset
    print("Checking dataset...")

    if not os.path.exists(DATA_PATH):

        print("ERROR: Dataset was not found.")
        print("Expected location:")
        print(DATA_PATH)

        return

    print("Dataset found!")
    print(DATA_PATH)

    # Load and split data
    print("Preparing data...")

    X_train, X_test, y_train, y_test = prepare_data(
        DATA_PATH
    )

    print("Data prepared successfully.")

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # Create preprocessing
    print("Creating preprocessing pipeline...")

    preprocessor = create_preprocessor()

    # Create Decision Tree
    model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        class_weight="balanced"
    )

    # Combine preprocessing + model
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    # Train
    print("Training Decision Tree...")

    pipeline.fit(
        X_train,
        y_train
    )

    print("Training completed!")

    # Prediction
    print("Making predictions...")

    predictions = pipeline.predict(
        X_test
    )

    probabilities = pipeline.predict_proba(
        X_test
    )[:, 1]

    # Metrics
    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    auc = roc_auc_score(
        y_test,
        probabilities
    )

    # Display results
    print()
    print("====================================")
    print("       DECISION TREE RESULTS")
    print("====================================")

    print(f"Accuracy    : {accuracy:.4f}")
    print(f"Precision   : {precision:.4f}")
    print(f"Recall      : {recall:.4f}")
    print(f"F1 Score    : {f1:.4f}")
    print(f"ROC-AUC     : {auc:.4f}")

    print()
    print("Confusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

    # Create models directory
    print()
    print("Creating models folder...")

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    # Save model
    print("Saving trained model...")

    joblib.dump(
        pipeline,
        MODEL_PATH
    )

    print()
    print("====================================")
    print("   MODEL SAVED SUCCESSFULLY!")
    print("====================================")

    print(
        f"Model location:\n{MODEL_PATH}"
    )


# --------------------------------------------------
# RUN PROGRAM
# --------------------------------------------------

if __name__ == "__main__":

    train_decision_tree()
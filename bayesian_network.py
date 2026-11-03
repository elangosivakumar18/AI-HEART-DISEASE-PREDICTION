import os
import pandas as pd
import joblib

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator


# ==========================================================
# PROJECT PATHS
# ==========================================================

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_DIR = os.path.dirname(
    CURRENT_DIR
)

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
    "bayesian_network.pkl"
)


# ==========================================================
# BAYESIAN NETWORK FEATURES
# ==========================================================

BN_FEATURES = [
    "age",
    "sex",
    "chest pain type",
    "max heart rate",
    "exercise induced angina",
    "oldpeak",
    "major vessels",
    "thal"
]

TARGET = "heart disease"


# ==========================================================
# PREPARE DATA
# ==========================================================

def prepare_bayesian_data(df):

    df = df.copy()

    print()
    print("Preparing Bayesian Network data...")


    # ------------------------------------------------------
    # CLEAN COLUMN NAMES
    # ------------------------------------------------------

    df.columns = (
        df.columns
        .str.strip()
    )


    # ------------------------------------------------------
    # CHECK REQUIRED COLUMNS
    # ------------------------------------------------------

    required_columns = [
        "age",
        "sex",
        "chest pain type",
        "max heart rate",
        "exercise induced angina",
        "oldpeak",
        "major vessels",
        "thal",
        "heart disease"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing columns in dataset:\n"
            + str(missing_columns)
        )


    # ------------------------------------------------------
    # TARGET
    # ------------------------------------------------------

    df[TARGET] = pd.to_numeric(
        df[TARGET],
        errors="coerce"
    )


    # Original dataset:
    #
    # 1 = No heart disease
    # 2 = Heart disease
    #
    # Bayesian Network:
    #
    # 0 = No disease
    # 1 = Disease

    df[TARGET] = (
        df[TARGET]
        .map({
            1: 0,
            2: 1
        })
    )


    # Remove invalid target rows

    df = df.dropna(
        subset=[TARGET]
    )


    # ------------------------------------------------------
    # AGE
    # ------------------------------------------------------

    df["age"] = pd.cut(

        df["age"],

        bins=[
            0,
            40,
            60,
            100
        ],

        labels=[
            "young",
            "middle",
            "old"
        ],

        include_lowest=True
    )


    # ------------------------------------------------------
    # MAX HEART RATE
    # ------------------------------------------------------

    df["max heart rate"] = pd.cut(

        df["max heart rate"],

        bins=[
            0,
            100,
            150,
            250
        ],

        labels=[
            "low",
            "medium",
            "high"
        ],

        include_lowest=True
    )


    # ------------------------------------------------------
    # OLDPEAK
    # ------------------------------------------------------

    df["oldpeak"] = pd.cut(

        df["oldpeak"],

        bins=[
            -1,
            1,
            2,
            10
        ],

        labels=[
            "low",
            "medium",
            "high"
        ],

        include_lowest=True
    )


    # ------------------------------------------------------
    # REMOVE MISSING VALUES
    # ------------------------------------------------------

    df = df.dropna(
        subset=BN_FEATURES + [TARGET]
    )


    # ------------------------------------------------------
    # CONVERT BN FEATURES TO STRING
    # ------------------------------------------------------

    for column in BN_FEATURES:

        df[column] = (
            df[column]
            .astype(str)
            .str.strip()
        )


    # ------------------------------------------------------
    # TARGET TO INTEGER
    # ------------------------------------------------------

    df[TARGET] = (
        df[TARGET]
        .astype(int)
    )


    # ------------------------------------------------------
    # KEEP ONLY REQUIRED COLUMNS
    # ------------------------------------------------------

    df = df[
        BN_FEATURES + [TARGET]
    ]


    print()
    print(
        "Final Bayesian Network columns:"
    )

    print(
        df.columns.tolist()
    )


    print()
    print(
        "Final dataset shape:",
        df.shape
    )


    return df


# ==========================================================
# CREATE NETWORK
# ==========================================================

def create_bayesian_network():

    print()
    print(
        "Creating Bayesian Network structure..."
    )


    # ------------------------------------------------------
    # IMPORTANT:
    #
    # Heart disease is the parent.
    #
    # This creates a Naive-Bayes-style Bayesian Network.
    #
    # Instead of estimating:
    #
    # P(heart disease | every combination of features)
    #
    # we estimate:
    #
    # P(heart disease)
    #
    # and:
    #
    # P(feature | heart disease)
    #
    # This is much more stable for a small dataset.
    # ------------------------------------------------------

    edges = [

        (
            TARGET,
            "age"
        ),

        (
            TARGET,
            "sex"
        ),

        (
            TARGET,
            "chest pain type"
        ),

        (
            TARGET,
            "max heart rate"
        ),

        (
            TARGET,
            "exercise induced angina"
        ),

        (
            TARGET,
            "oldpeak"
        ),

        (
            TARGET,
            "major vessels"
        ),

        (
            TARGET,
            "thal"
        )
    ]


    model = DiscreteBayesianNetwork(
        edges
    )


    return model


# ==========================================================
# TRAIN BAYESIAN NETWORK
# ==========================================================

def train_bayesian_network():

    print()
    print("==============================================")
    print("      BAYESIAN NETWORK TRAINING")
    print("==============================================")


    # ------------------------------------------------------
    # CHECK DATASET
    # ------------------------------------------------------

    if not os.path.exists(DATA_PATH):

        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}"
        )


    print()
    print(
        "Dataset found:"
    )

    print(
        DATA_PATH
    )


    # ------------------------------------------------------
    # LOAD DATA
    # ------------------------------------------------------

    print()
    print(
        "Loading dataset..."
    )


    df = pd.read_csv(
        DATA_PATH
    )


    print(
        "Original dataset shape:",
        df.shape
    )


    # ------------------------------------------------------
    # PREPARE DATA
    # ------------------------------------------------------

    df = prepare_bayesian_data(
        df
    )


    # ------------------------------------------------------
    # TARGET DISTRIBUTION
    # ------------------------------------------------------

    print()
    print(
        "Heart disease distribution:"
    )

    print(
        df[TARGET]
        .value_counts()
        .sort_index()
    )


    # ------------------------------------------------------
    # CREATE NETWORK
    # ------------------------------------------------------

    model = create_bayesian_network()


    print()
    print(
        "Network nodes:"
    )

    print(
        list(model.nodes())
    )


    print()
    print(
        "Network edges:"
    )

    print(
        list(model.edges())
    )


    # ------------------------------------------------------
    # BAYESIAN PARAMETER LEARNING
    # ------------------------------------------------------

    print()
    print(
        "Training Bayesian Network..."
    )

    print(
        "Using BayesianEstimator with BDeu smoothing..."
    )


    estimator = BayesianEstimator(
        model,
        df
    )


    # ------------------------------------------------------
    # LEARN CPDs
    # ------------------------------------------------------

    cpds = estimator.get_parameters(

        prior_type="BDeu",

        equivalent_sample_size=10,

        n_jobs=1
    )


    model.add_cpds(
        *cpds
    )


    # ------------------------------------------------------
    # VALIDATE MODEL
    # ------------------------------------------------------

    print()
    print(
        "Checking Bayesian Network..."
    )


    if not model.check_model():

        raise ValueError(
            "Bayesian Network validation failed."
        )


    print(
        "Bayesian Network validation successful!"
    )


    # ------------------------------------------------------
    # DISPLAY CPDS
    # ------------------------------------------------------

    print()
    print(
        "Number of CPDs:"
    )

    print(
        len(model.get_cpds())
    )


    print()
    print(
        "=============================================="
    )

    print(
        "HEART DISEASE CPD"
    )

    print(
        "=============================================="
    )


    print(
        model.get_cpds(
            TARGET
        )
    )


    # ------------------------------------------------------
    # DISPLAY ALL CPDS
    # ------------------------------------------------------

    print()
    print(
        "=============================================="
    )

    print(
        "LEARNED CPDs"
    )

    print(
        "=============================================="
    )


    for cpd in model.get_cpds():

        print()
        print(
            cpd
        )


    # ------------------------------------------------------
    # SAVE MODEL
    # ------------------------------------------------------

    print()
    print(
        "Saving Bayesian Network..."
    )


    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )


    joblib.dump(
        model,
        MODEL_PATH
    )


    print()
    print(
        "=============================================="
    )

    print(
        " BAYESIAN NETWORK TRAINING COMPLETE"
    )

    print(
        "=============================================="
    )


    print()
    print(
        "Model saved successfully:"
    )

    print(
        MODEL_PATH
    )


    return model


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    train_bayesian_network()

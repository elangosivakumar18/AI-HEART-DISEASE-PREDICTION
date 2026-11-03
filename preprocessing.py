import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


# ==========================================================
# ACTUAL COLUMNS IN YOUR DATASET
# ==========================================================

CATEGORICAL_FEATURES = [
    "sex",
    "chest pain type",
    "fasting blood sugar",
    "resting electrocardiographic results",
    "exercise induced angina",
    "ST segment",
    "major vessels",
    "thal"
]

NUMERICAL_FEATURES = [
    "age",
    "resting blood pressure",
    "serum cholestoral",
    "max heart rate",
    "oldpeak"
]

FEATURES = (
    CATEGORICAL_FEATURES +
    NUMERICAL_FEATURES
)

TARGET = "heart disease"


# ==========================================================
# LOAD DATA
# ==========================================================

def load_data(file_path):

    print("Loading dataset...")

    df = pd.read_csv(file_path)

    # Remove accidental spaces from column names
    df.columns = df.columns.str.strip()

    print("Dataset loaded successfully.")
    print("Dataset shape:", df.shape)

    print("Columns:")
    print(df.columns.tolist())

    return df


# ==========================================================
# PREPARE TARGET
# ==========================================================

def prepare_target(df):

    df = df.copy()

    print("Preparing target column...")

    print(
        "Heart disease values:",
        df[TARGET].unique()
    )

    # Convert target to numeric
    #
    # This handles datasets where the target is already
    # 0/1, 1/2, or common text representations.

    if df[TARGET].dtype == "object":

        df[TARGET] = (
            df[TARGET]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({
                "no": 0,
                "yes": 1,
                "negative": 0,
                "positive": 1,
                "absence": 0,
                "presence": 1,
                "0": 0,
                "1": 1,
                "2": 2,
            })
        )

    else:

        df[TARGET] = pd.to_numeric(
            df[TARGET],
            errors="coerce"
        )

    numeric_values = sorted(
        set(df[TARGET].dropna().astype(int).unique())
    )

    # Standardize common heart-disease encodings to the binary form
    # expected by the downstream classifiers.
    if set(numeric_values).issubset({0, 1}):
        pass
    elif set(numeric_values).issubset({1, 2}):
        df[TARGET] = df[TARGET].map({1: 0, 2: 1})
    elif set(numeric_values).issubset({0, 1, 2}):
        df[TARGET] = df[TARGET].map({0: 0, 1: 1, 2: 1})

    return df


# ==========================================================
# CREATE PREPROCESSING PIPELINE
# ==========================================================

def create_preprocessor():

    # Numerical data
    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )


    # Categorical data
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )


    # Combine both
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                NUMERICAL_FEATURES
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES
            )
        ]
    )

    return preprocessor


# ==========================================================
# PREPARE TRAINING AND TESTING DATA
# ==========================================================

def prepare_data(file_path):

    # Load data
    df = load_data(file_path)


    # Prepare target
    df = prepare_target(df)


    # Check required columns
    required_columns = FEATURES + [TARGET]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]


    if missing_columns:

        raise ValueError(
            "These required columns are missing:\n"
            f"{missing_columns}"
        )


    # Remove rows where target is missing
    df = df.dropna(
        subset=[TARGET]
    )


    # Input features
    X = df[FEATURES]


    # Target
    y = df[TARGET].astype(int)


    print()
    print("Features:")
    print(FEATURES)

    print()
    print("Target:")
    print(TARGET)

    print()
    print("Target distribution:")
    print(y.value_counts())


    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=0.20,

        random_state=42,

        stratify=y
    )


    print()
    print("Training data:", X_train.shape)

    print("Testing data:", X_test.shape)


    return (
        X_train,
        X_test,
        y_train,
        y_test
    )
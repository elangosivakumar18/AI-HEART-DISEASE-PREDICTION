import os
import joblib
import pandas as pd

from pgmpy.inference import VariableElimination


# ==========================================================
# PROJECT PATHS
# ==========================================================

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_DIR = os.path.dirname(
    CURRENT_DIR
)

DECISION_TREE_PATH = os.path.join(
    PROJECT_DIR,
    "models",
    "decision_tree.pkl"
)

BAYESIAN_NETWORK_PATH = os.path.join(
    PROJECT_DIR,
    "models",
    "bayesian_network.pkl"
)


# ==========================================================
# LOAD MODELS
# ==========================================================

def load_models():

    if not os.path.exists(DECISION_TREE_PATH):

        raise FileNotFoundError(
            "Decision Tree model not found.\n"
            f"Expected location: {DECISION_TREE_PATH}"
        )

    if not os.path.exists(BAYESIAN_NETWORK_PATH):

        raise FileNotFoundError(
            "Bayesian Network model not found.\n"
            f"Expected location: {BAYESIAN_NETWORK_PATH}"
        )

    decision_tree = joblib.load(
        DECISION_TREE_PATH
    )

    bayesian_network = joblib.load(
        BAYESIAN_NETWORK_PATH
    )

    return decision_tree, bayesian_network


# ==========================================================
# PATIENT DATA NORMALIZATION
# ==========================================================

def normalize_patient_data(patient_data):
    """
    Normalizes patient data to the standard Statlog encoding used by both
    the Decision Tree and Bayesian Network models:
      - chest pain type: 1 (Typical Angina), 2 (Atypical Angina), 3 (Non-anginal Pain), 4 (Asymptomatic)
      - ST segment: 1 (Upsloping), 2 (Flat), 3 (Downsloping)
      - thal: 3 (Normal), 6 (Fixed Defect), 7 (Reversible Defect)
      - sex: 0 (Female), 1 (Male)
      - fasting blood sugar: 0, 1
      - resting electrocardiographic results: 0, 1, 2
      - exercise induced angina: 0, 1
      - major vessels: 0, 1, 2, 3
      - age, resting blood pressure, serum cholestoral, max heart rate, oldpeak: numeric
    """
    data = dict(patient_data)

    # 1. Chest pain type (handles 0..3 Cleveland -> 1..4 Statlog)
    cp = data.get("chest pain type")
    if cp is not None:
        try:
            cp_int = int(float(cp))
            if cp_int == 0:
                data["chest pain type"] = 1
            elif cp_int in (1, 2, 3, 4):
                data["chest pain type"] = cp_int
            else:
                data["chest pain type"] = 4
        except (ValueError, TypeError):
            data["chest pain type"] = 4
    else:
        data["chest pain type"] = 2

    # 2. Thal (handles 1, 2, 3 Cleveland -> 3, 6, 7 Statlog)
    thal = data.get("thal")
    if thal is not None:
        try:
            thal_int = int(float(thal))
            if thal_int == 1:
                data["thal"] = 3
            elif thal_int == 2:
                data["thal"] = 6
            elif thal_int == 3:
                data["thal"] = 3
            elif thal_int in (6, 7):
                data["thal"] = thal_int
            else:
                data["thal"] = 3
        except (ValueError, TypeError):
            data["thal"] = 3
    else:
        data["thal"] = 3

    # 3. ST segment (handles 0..2 Cleveland -> 1..3 Statlog)
    st = data.get("ST segment")
    if st is not None:
        try:
            st_int = int(float(st))
            if st_int == 0:
                data["ST segment"] = 1
            elif st_int in (1, 2, 3):
                data["ST segment"] = st_int
            else:
                data["ST segment"] = 1
        except (ValueError, TypeError):
            data["ST segment"] = 1
    else:
        data["ST segment"] = 1

    # 4. Sex (0 or 1)
    sex = data.get("sex")
    if sex is not None:
        try:
            data["sex"] = 0 if str(sex).strip().lower() in ("0", "f", "female") else 1
        except Exception:
            data["sex"] = 1
    else:
        data["sex"] = 1

    # 5. Fasting blood sugar (0 or 1)
    fbs = data.get("fasting blood sugar")
    if fbs is not None:
        try:
            data["fasting blood sugar"] = 1 if int(float(fbs)) >= 1 else 0
        except Exception:
            data["fasting blood sugar"] = 0
    else:
        data["fasting blood sugar"] = 0

    # 6. Exercise induced angina (0 or 1)
    exang = data.get("exercise induced angina")
    if exang is not None:
        try:
            data["exercise induced angina"] = 1 if int(float(exang)) >= 1 else 0
        except Exception:
            data["exercise induced angina"] = 0
    else:
        data["exercise induced angina"] = 0

    # 7. Resting ECG (0, 1, 2)
    ecg = data.get("resting electrocardiographic results")
    if ecg is not None:
        try:
            ecg_int = int(float(ecg))
            data["resting electrocardiographic results"] = ecg_int if ecg_int in (0, 1, 2) else 0
        except Exception:
            data["resting electrocardiographic results"] = 0
    else:
        data["resting electrocardiographic results"] = 0

    # 8. Major vessels (0, 1, 2, 3)
    ca = data.get("major vessels")
    if ca is not None:
        try:
            ca_int = int(float(ca))
            data["major vessels"] = max(0, min(3, ca_int))
        except Exception:
            data["major vessels"] = 0
    else:
        data["major vessels"] = 0

    # Numeric fields
    num_defaults = [
        ("age", 55.0),
        ("resting blood pressure", 120.0),
        ("serum cholestoral", 200.0),
        ("max heart rate", 150.0),
        ("oldpeak", 1.0),
    ]
    for num_col, default in num_defaults:
        val = data.get(num_col)
        try:
            data[num_col] = float(val) if val is not None else default
        except (ValueError, TypeError):
            data[num_col] = default

    return data


# ==========================================================
# DECISION TREE PREDICTION
# ==========================================================

def predict_decision_tree(patient_data):

    decision_tree, _ = load_models()
    clean_data = normalize_patient_data(patient_data)

    patient_df = pd.DataFrame(
        [clean_data]
    )

    raw_prediction = int(decision_tree.predict(
        patient_df
    )[0])

    probabilities = decision_tree.predict_proba(
        patient_df
    )[0]

    class_labels = [int(label) for label in decision_tree.classes_]
    disease_label = 1 if 1 in class_labels else 2 if 2 in class_labels else 0
    disease_index = class_labels.index(disease_label)
    probability = float(probabilities[disease_index]) if len(probabilities) > 1 else float(probabilities[0])

    prediction = int(raw_prediction == disease_label)

    return {
        "prediction": prediction,
        "probability": probability
    }


# ==========================================================
# BAYESIAN NETWORK DATA PREPARATION
# ==========================================================

def prepare_bayesian_evidence(patient_data):

    """
    Prepare ONLY the features that exist in the
    trained Bayesian Network.

    Bayesian Network features:
        age, sex, chest pain type, max heart rate,
        exercise induced angina, oldpeak, major vessels, thal
    """

    clean_data = normalize_patient_data(patient_data)
    evidence = {}

    # AGE
    age = clean_data["age"]
    if age <= 40:
        evidence["age"] = "young"
    elif age <= 60:
        evidence["age"] = "middle"
    else:
        evidence["age"] = "old"

    # SEX
    evidence["sex"] = str(clean_data["sex"])

    # CHEST PAIN TYPE
    evidence["chest pain type"] = str(clean_data["chest pain type"])

    # MAXIMUM HEART RATE
    max_heart_rate = clean_data["max heart rate"]
    if max_heart_rate <= 100:
        evidence["max heart rate"] = "low"
    elif max_heart_rate <= 150:
        evidence["max heart rate"] = "medium"
    else:
        evidence["max heart rate"] = "high"

    # EXERCISE INDUCED ANGINA
    evidence["exercise induced angina"] = str(clean_data["exercise induced angina"])

    # OLDPEAK
    oldpeak = clean_data["oldpeak"]
    if oldpeak <= 1:
        evidence["oldpeak"] = "low"
    elif oldpeak <= 2:
        evidence["oldpeak"] = "medium"
    else:
        evidence["oldpeak"] = "high"

    # MAJOR VESSELS
    evidence["major vessels"] = str(clean_data["major vessels"])

    # THAL
    evidence["thal"] = str(clean_data["thal"])

    return evidence


# ==========================================================
# BAYESIAN NETWORK PREDICTION
# ==========================================================

def predict_bayesian_network(patient_data):

    _, bayesian_network = load_models()

    if "heart disease" not in bayesian_network.nodes():
        raise ValueError(
            "The Bayesian Network does not contain 'heart disease' as a node."
        )

    evidence = prepare_bayesian_evidence(
        patient_data
    )

    try:
        inference = VariableElimination(
            bayesian_network
        )

        result = inference.query(
            variables=["heart disease"],
            evidence=evidence,
            show_progress=False
        )

        states = result.state_names[
            "heart disease"
        ]
        probabilities = result.values

        probability_no_disease = 0.0
        probability_disease = 0.0

        for state, probability in zip(
            states,
            probabilities
        ):
            if int(state) == 0:
                probability_no_disease = float(probability)
            elif int(state) == 1:
                probability_disease = float(probability)

        prediction = 1 if probability_disease >= 0.5 else 0

        return {
            "prediction": prediction,
            "probability": probability_disease,
            "no_disease_probability": probability_no_disease,
            "disease_probability": probability_disease
        }

    except Exception as error:
        print(f"Warning: Bayesian Network inference issue: {error}. Falling back to baseline prior.")
        return {
            "prediction": 0,
            "probability": 0.446,
            "no_disease_probability": 0.554,
            "disease_probability": 0.446
        }


# ==========================================================
# COMBINED PREDICTION
# ==========================================================

def predict(patient_data):

    # ------------------------------------------------------
    # Decision Tree
    # ------------------------------------------------------

    decision_tree_result = (
        predict_decision_tree(
            patient_data
        )
    )


    # ------------------------------------------------------
    # Bayesian Network
    # ------------------------------------------------------

    bayesian_result = (
        predict_bayesian_network(
            patient_data
        )
    )


    # ------------------------------------------------------
    # COMBINE RESULTS
    # ------------------------------------------------------

    return {

        "decision_tree":
            decision_tree_result,

        "bayesian_network":
            bayesian_result
    }


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    print()
    print("====================================")
    print("       TESTING PREDICTOR")
    print("====================================")


    # ------------------------------------------------------
    # TEST PATIENT
    # ------------------------------------------------------

    patient = {

        "age": 55,

        "sex": 1,

        "chest pain type": 2,

        "resting blood pressure": 140,

        "serum cholestoral": 250,

        "fasting blood sugar": 0,

        "resting electrocardiographic results": 1,

        "max heart rate": 150,

        "exercise induced angina": 0,

        "oldpeak": 1.5,

        "ST segment": 2,

        "major vessels": 0,

        "thal": 3
    }


    # ------------------------------------------------------
    # RUN PREDICTION
    # ------------------------------------------------------

    print()
    print("Testing patient data...")


    result = predict(
        patient
    )


    # ------------------------------------------------------
    # DISPLAY RESULTS
    # ------------------------------------------------------

    print()
    print("====================================")
    print("           PREDICTION RESULT")
    print("====================================")


    print()
    print("Decision Tree:")

    print(
        result["decision_tree"]
    )


    print()
    print("Bayesian Network:")

    print(
        result["bayesian_network"]
    )


    # ------------------------------------------------------
    # HUMAN-READABLE RESULT
    # ------------------------------------------------------

    print()
    print("====================================")
    print("        FINAL PREDICTION")
    print("====================================")


    dt_probability = result[
        "decision_tree"
    ]["probability"]


    bn_probability = result[
        "bayesian_network"
    ]["probability"]


    print()
    print(
        f"Decision Tree probability: "
        f"{dt_probability:.2%}"
    )


    print(
        f"Bayesian Network probability: "
        f"{bn_probability:.2%}"
    )


    # ------------------------------------------------------
    # Simple combined probability
    # ------------------------------------------------------

    combined_probability = (
        dt_probability + bn_probability
    ) / 2


    print(
        f"Combined probability: "
        f"{combined_probability:.2%}"
    )


    if combined_probability >= 0.5:

        print()
        print(
            "Result: Higher predicted risk"
        )

    else:

        print()
        print(
            "Result: Lower predicted risk"
        )


    print()
    print("==============================")
    print("       PREDICTOR TEST COMPLETE")
    print("==========================================")
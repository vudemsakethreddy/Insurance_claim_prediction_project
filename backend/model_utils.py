import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Dict, Any, List
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
    classification_report  # ✅ ADDED
)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Optional models
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")
STATIC_PLOTS_DIR = os.path.join(BASE_DIR, "static", "plots")

BEST_MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")
META_PATH = os.path.join(MODEL_DIR, "meta.json")

TARGET_COL = "insuranceclaim"


def _ensure_dirs():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(STATIC_PLOTS_DIR, exist_ok=True)


def _resolve_dataset_path() -> str:
    preferred = os.path.join(DATA_DIR, "dataset.csv")
    if os.path.exists(preferred):
        return preferred

    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Data folder not found: {DATA_DIR}")

    csv_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(f"No CSV found in: {DATA_DIR}")

    return os.path.join(DATA_DIR, csv_files[0])


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
    return df


def _iqr_cap(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    return s.clip(lower=low, upper=high)


def _feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Safe numeric conversion
    for col in ["age", "bmi", "children", "charges"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Map common string categories (still fine if already numeric)
    if "gender" in df.columns and df["gender"].dtype == "object":
        df["gender"] = df["gender"].astype(str).str.lower().map({"female": 0, "male": 1})

    if "smoker" in df.columns and df["smoker"].dtype == "object":
        df["smoker"] = df["smoker"].astype(str).str.lower().map({"no": 0, "yes": 1})

    # Median imputation numeric
    for col in ["age", "bmi", "children", "charges"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Outlier cap (PDF: bmi and charges)
    if "bmi" in df.columns:
        df["bmi"] = _iqr_cap(df["bmi"])
    if "charges" in df.columns:
        df["charges"] = _iqr_cap(df["charges"])

    # Engineered features (PDF)
    if "smoker" in df.columns and "bmi" in df.columns:
        df["smoker_bmi"] = pd.to_numeric(df["smoker"], errors="coerce").fillna(0) * df["bmi"]

    if "age" in df.columns:
        df["age_group"] = pd.cut(
            df["age"],
            bins=[0, 25, 35, 45, 55, 65, 200],
            labels=["<25", "25-35", "35-45", "45-55", "55-65", "65+"],
            include_lowest=True
        )

    if "bmi" in df.columns:
        df["bmi_category"] = pd.cut(
            df["bmi"],
            bins=[0, 18.5, 25, 30, 200],
            labels=["Under", "Normal", "Over", "Obese"],
            include_lowest=True
        )

    if "charges" in df.columns:
        df["charges_log"] = np.log1p(df["charges"])

    return df


def _save_fig(filename: str):
    path = os.path.join(STATIC_PLOTS_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _eda_and_save_plots(df: pd.DataFrame):
    if TARGET_COL in df.columns:
        vc = df[TARGET_COL].value_counts(dropna=False)
        plt.figure()
        plt.bar(vc.index.astype(str), vc.values)
        plt.title("Target Distribution (insuranceclaim)")
        plt.xlabel("Class")
        plt.ylabel("Count")
        _save_fig("01_target_distribution.png")

    for col in ["age", "bmi", "charges", "charges_log"]:
        if col in df.columns:
            plt.figure()
            plt.hist(pd.to_numeric(df[col], errors="coerce").dropna(), bins=30)
            plt.title(f"Histogram: {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            _save_fig(f"02_hist_{col}.png")

    if "smoker" in df.columns and TARGET_COL in df.columns:
        tmp = df.copy()
        tmp["smoker"] = pd.to_numeric(tmp["smoker"], errors="coerce")
        grp = tmp.groupby("smoker")[TARGET_COL].mean().dropna()
        plt.figure()
        plt.bar(grp.index.astype(str), grp.values)
        plt.title("Mean Claim Rate by Smoker (0=No, 1=Yes)")
        plt.xlabel("Smoker")
        plt.ylabel("Mean Claim Rate")
        _save_fig("03_smoker_vs_claim.png")

    if "region" in df.columns and TARGET_COL in df.columns:
        grp = df.groupby("region")[TARGET_COL].mean().sort_values(ascending=False)
        plt.figure()
        plt.bar(grp.index.astype(str), grp.values)
        plt.title("Mean Claim Rate by Region")
        plt.xlabel("Region")
        plt.ylabel("Mean Claim Rate")
        plt.xticks(rotation=30, ha="right")
        _save_fig("04_region_vs_claim.png")

    num_df = df.select_dtypes(include=[np.number])
    if len(num_df.columns) >= 2:
        corr = num_df.corr()
        plt.figure(figsize=(7, 6))
        plt.imshow(corr.values, aspect="auto")
        plt.colorbar()
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
        plt.yticks(range(len(corr.index)), corr.index)
        plt.title("Correlation Heatmap (Numeric)")
        _save_fig("05_corr_heatmap.png")

    for col in ["bmi", "charges"]:
        if col in df.columns:
            plt.figure()
            plt.boxplot(pd.to_numeric(df[col], errors="coerce").dropna(), vert=True)
            plt.title(f"Boxplot: {col}")
            _save_fig(f"06_boxplot_{col}.png")


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = []
    categorical_features = []
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            numeric_features.append(c)
        else:
            categorical_features.append(c)

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scaler", StandardScaler())]), numeric_features),
            ("cat", Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_features),
        ],
        remainder="drop",
    )


def _evaluate(pipe, X_test, y_test) -> Dict[str, Any]:
    pred = pipe.predict(X_test)

    proba = None
    if hasattr(pipe, "predict_proba"):
        try:
            proba = pipe.predict_proba(X_test)[:, 1]
        except Exception:
            proba = None

    if proba is None and hasattr(pipe, "decision_function"):
        try:
            scores = pipe.decision_function(X_test)
            proba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        except Exception:
            proba = None

    out = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "pred": pred,
        "proba": proba,
        "roc_auc": float(roc_auc_score(y_test, proba)) if proba is not None else float("nan"),
    }
    return out


def train_all_models(test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
    _ensure_dirs()
    data_path = _resolve_dataset_path()

    df = pd.read_csv(data_path)
    df = _standardize_columns(df)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found. Found: {df.columns.tolist()}")

    df = df.drop_duplicates().reset_index(drop=True)
    df = _feature_engineer(df)
    _eda_and_save_plots(df)

    y = pd.to_numeric(df[TARGET_COL], errors="coerce").fillna(0).astype(int)
    X = df.drop(columns=[TARGET_COL])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    preprocessor = _build_preprocessor(X_train)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=250, random_state=random_state),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        # Keep SVM disabled if it was slow
        # "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=random_state),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=350, learning_rate=0.05, max_depth=4,
            subsample=0.9, colsample_bytree=0.9,
            random_state=random_state, eval_metric="logloss",
        )
    if HAS_LGBM:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=500, learning_rate=0.05, random_state=random_state
        )

    best_name, best_pipe, best_auc = None, None, -1.0
    results: List[Dict[str, Any]] = []
    best_detail = None

    for name, clf in models.items():
        print(f"\nTraining model: {name} ...")

        pipe = ImbPipeline(steps=[
            ("preprocess", preprocessor),
            ("smote", SMOTE(random_state=random_state, k_neighbors=3)),
            ("model", clf),
        ])

        pipe.fit(X_train, y_train)
        print(f"Finished model: {name}")

        # ✅ THIS PRINTS OUTPUT LIKE YOUR SCREENSHOT
        y_pred = pipe.predict(X_test)
        print(f"\n===== {name} =====")
        print(classification_report(y_test, y_pred))

        detail = _evaluate(pipe, X_test, y_test)

        results.append({
            "model": name,
            "accuracy": round(detail["accuracy"], 6),
            "precision": round(detail["precision"], 6),
            "recall": round(detail["recall"], 6),
            "f1": round(detail["f1"], 6),
            "roc_auc": None if np.isnan(detail["roc_auc"]) else round(detail["roc_auc"], 6),
        })

        if (not np.isnan(detail["roc_auc"])) and detail["roc_auc"] > best_auc:
            best_auc = float(detail["roc_auc"])
            best_name = name
            best_pipe = pipe
            best_detail = detail

    joblib.dump(best_pipe, BEST_MODEL_PATH)

    comp = pd.DataFrame(results)
    comp_sorted = comp.sort_values(by="roc_auc", ascending=False, na_position="last")
    if len(comp_sorted) > 0:
        plt.figure(figsize=(8, 4))
        plt.plot(comp_sorted["model"], comp_sorted["roc_auc"], marker="o")
        plt.title("Model Comparison (ROC-AUC)")
        plt.xlabel("Model")
        plt.ylabel("ROC-AUC")
        plt.xticks(rotation=30, ha="right")
        _save_fig("07_model_comparison_roc_auc.png")

    if best_detail is not None:
        cm = confusion_matrix(y_test, best_detail["pred"])
        plt.figure()
        plt.imshow(cm, aspect="auto")
        plt.colorbar()
        plt.title(f"Confusion Matrix - {best_name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        for (i, j), v in np.ndenumerate(cm):
            plt.text(j, i, str(v), ha="center", va="center")
        _save_fig("08_confusion_matrix_best.png")

        if best_detail["proba"] is not None:
            fpr, tpr, _ = roc_curve(y_test, best_detail["proba"])
            plt.figure()
            plt.plot(fpr, tpr, label=f"{best_name} (AUC={best_auc:.3f})")
            plt.plot([0, 1], [0, 1], linestyle="--")
            plt.title("ROC Curve (Best Model)")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend()
            _save_fig("09_roc_curve_best.png")

    meta = {
        "dataset_used": os.path.basename(data_path),
        "best_model": best_name,
        "best_roc_auc": best_auc,
        "results": results,
        "plots_dir": STATIC_PLOTS_DIR,
        "note": "; ".join([
            ("XGBoost not installed" if not HAS_XGB else ""),
            ("LightGBM not installed" if not HAS_LGBM else "")
        ]).strip("; ").strip()
    }

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return meta


def predict_one(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not os.path.exists(BEST_MODEL_PATH):
        raise FileNotFoundError("Model not found. Train first.")

    pipe = joblib.load(BEST_MODEL_PATH)

    df = pd.DataFrame([payload])
    df = _standardize_columns(df)
    df = _feature_engineer(df)

    pred = int(pipe.predict(df)[0])

    proba = None
    if hasattr(pipe, "predict_proba"):
        try:
            proba = float(pipe.predict_proba(df)[0, 1])
        except Exception:
            proba = None

    return {"prediction": pred, "probability": proba}


# print("Starting training...")
# _result = train_all_models()
# print("Training finished!")

# print(f"Best Model: {_result['best_model']} | Best ROC-AUC: {_result['best_roc_auc']}")



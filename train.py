"""
Ejemplo variando hiperparámetros:
    python train.py --c_param 0.1 --max_iter 200 --solver liblinear --seed 7
"""
import argparse
import json
import random
import tempfile
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "HeartDisease"
NUM_FEATURES = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
BIN_FEATURES = ["FastingBS", "Cholesterol_missing"]
CAT_FEATURES = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
CATEGORICAL_RAW = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # Hiperparámetros del modelo / preprocesamiento
    p.add_argument("--c_param", "--C", dest="C", type=float, default=1.0,
                   help="Inverso de la fuerza de regularización L2 (default: 1.0)")
    p.add_argument("--max_iter", "--max-iter", dest="max_iter", type=int, default=1000,
                   help="Iteraciones máximas del solver (default: 1000)")
    p.add_argument("--solver", choices=["lbfgs", "liblinear", "newton-cg", "saga"],
                   default="lbfgs", help="Algoritmo de optimización (default: lbfgs)")
    p.add_argument("--class-weight", choices=["none", "balanced"], default="none",
                   help="Ponderación de clases (default: none)")
    p.add_argument("--imputer-strategy", choices=["median", "mean"], default="median",
                   help="Estrategia de imputación numérica (default: median)")
    # Reproducibilidad y partición
    p.add_argument("--seed", type=int, default=42,
                   help="Semilla aleatoria para partición, modelo y librerías (default: 42)")
    p.add_argument("--test-size", type=float, default=0.2,
                   help="Proporción del conjunto de prueba (default: 0.2)")
    # Datos y MLflow
    p.add_argument("--data-path", default="heart.csv", help="Ruta del CSV (default: heart.csv)")
    p.add_argument("--tracking-uri", default="sqlite:///mlflow.db",
                   help="URI de tracking de MLflow (default: sqlite:///mlflow.db)")
    p.add_argument("--experiment-name", default="heart_disease_logreg_hiperparametros",
                   help="Nombre del experimento de MLflow")
    p.add_argument("--run-name", default=None, help="Nombre de la run (default: automático)")
    p.add_argument("--variation", default="baseline",
                   help="Etiqueta del factor que se varía respecto al baseline (tag de MLflow)")
    return p.parse_args(argv)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def load_data(path):
    """Carga el CSV del dataset."""
    return pd.read_csv(path)


def clean_data(df):
    """Limpieza (mismas decisiones del notebook). Devuelve (df_limpio, decisiones, n_duplicados)."""
    decisiones = []
    df = df.copy()

    faltantes = int(df.isna().sum().sum())
    decisiones.append(f"Faltantes explícitos (NaN): {faltantes}. No requiere tratamiento directo.")

    n_chol0 = int((df["Cholesterol"] == 0).sum())
    n_bp0 = int((df["RestingBP"] == 0).sum())
    df["Cholesterol"] = df["Cholesterol"].replace(0, np.nan)
    df["RestingBP"] = df["RestingBP"].replace(0, np.nan)
    decisiones.append(
        f"Cholesterol = 0 en {n_chol0} filas y RestingBP = 0 en {n_bp0} fila(s): son faltantes "
        "ocultos. Se convierten a NaN y se imputan dentro del pipeline (ajustado solo con train).")

    df["Cholesterol_missing"] = df["Cholesterol"].isna().astype(int)
    decisiones.append("Se crea 'Cholesterol_missing' (1 si estaba en 0/NaN) para conservar la señal.")

    n_dup = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    decisiones.append(f"Duplicados encontrados y eliminados: {n_dup}.")

    for c in ["FastingBS", TARGET]:
        df[c] = df[c].astype(int)
    for c in CATEGORICAL_RAW:
        df[c] = df[c].astype("category")
    decisiones.append("Tipos: binarias a int; categóricas a 'category' (One-Hot en el pipeline).")
    decisiones.append("Outliers y Oldpeak negativo: se conservan (plausibles clínicamente); "
                      "se mitigan con StandardScaler y regularización L2.")
    return df, decisiones, n_dup


def split_data(df_clean, test_size, seed):
    """Partición estratificada train/test."""
    X = df_clean.drop(columns=[TARGET])
    y = df_clean[TARGET]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)

def build_preprocessor(imputer_strategy):
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy=imputer_strategy)),
                          ("sc", StandardScaler())]), NUM_FEATURES),
        ("bin", "passthrough", BIN_FEATURES),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CAT_FEATURES),
    ])


def build_model(args):
    class_weight = None if args.class_weight == "none" else args.class_weight
    clf = LogisticRegression(C=args.C, max_iter=args.max_iter, class_weight=class_weight,
                             solver=args.solver, random_state=args.seed)
    return Pipeline([("prep", build_preprocessor(args.imputer_strategy)), ("clf", clf)])


def build_dummy(args):
    return Pipeline([("prep", build_preprocessor(args.imputer_strategy)),
                     ("clf", DummyClassifier(strategy="most_frequent"))])


def evaluate(model, X_test, y_test):
    """Métricas de clasificación sobre el conjunto de prueba."""
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
    }
    return {k: float(v) for k, v in metrics.items()}, pred, proba


def run_training(args):
    """Pipeline completo sin MLflow: datos -> limpieza -> split -> entrenamiento -> evaluación."""
    warnings.filterwarnings("ignore")
    set_seed(args.seed)
    df = load_data(args.data_path)
    df_clean, decisiones, n_dup = clean_data(df)
    X_train, X_test, y_train, y_test = split_data(df_clean, args.test_size, args.seed)

    model = build_model(args).fit(X_train, y_train)
    dummy = build_dummy(args).fit(X_train, y_train)

    metrics, pred, proba = evaluate(model, X_test, y_test)
    metrics_dummy, _, _ = evaluate(dummy, X_test, y_test)
    metrics["accuracy_gain_vs_dummy"] = metrics["accuracy"] - metrics_dummy["accuracy"]
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y_test, pred).ravel())

    return {
        "model": model, "metrics": metrics, "metrics_dummy": metrics_dummy,
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "X_train": X_train, "X_test": X_test, "y_test": y_test,
        "pred": pred, "proba": proba, "decisiones": decisiones, "n_dup": n_dup,
    }

def save_artifacts(args, res, out_dir):
    """Genera los archivos que se registran como artefactos. Devuelve sus rutas."""
    out_dir = Path(out_dir)
    paths = []

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(res["y_test"], res["pred"],
                                            display_labels=["Sano", "Enfermo"], ax=ax)
    ax.set_title("Matriz de confusión (test)")
    p = out_dir / "matriz_confusion.png"
    fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig)
    paths.append(p)

    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(res["y_test"], res["proba"], ax=ax)
    ax.set_title("Curva ROC (test)")
    p = out_dir / "curva_roc.png"
    fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig)
    paths.append(p)

    p = out_dir / "metricas.json"
    p.write_text(json.dumps({"args": vars(args), "metrics": res["metrics"],
                             "metrics_dummy": res["metrics_dummy"],
                             "confusion": res["confusion"]}, indent=2, ensure_ascii=False),
                 encoding="utf-8")
    paths.append(p)

    p = out_dir / "decisiones_limpieza.txt"
    p.write_text("\n".join(res["decisiones"]), encoding="utf-8")
    paths.append(p)
    return paths


def default_run_name(args):
    return (f"C{args.C:g}_{args.solver}_it{args.max_iter}"
            f"_cw-{args.class_weight}_seed{args.seed}")


def log_to_mlflow(args, res):
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name=args.run_name or default_run_name(args)):
        mlflow.set_tag("variacion", args.variation)

        mlflow.log_param("modelo", "LogisticRegression")
        mlflow.log_param("C", args.C)
        mlflow.log_param("max_iter", args.max_iter)
        mlflow.log_param("solver", args.solver)
        mlflow.log_param("class_weight", args.class_weight)
        mlflow.log_param("imputer_strategy", args.imputer_strategy)
        mlflow.log_param("seed", args.seed)
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("n_train", len(res["X_train"]))
        mlflow.log_param("n_test", len(res["X_test"]))
        mlflow.log_param("duplicados_eliminados", res["n_dup"])

        for name, value in res["metrics"].items():
            mlflow.log_metric(name, value)
        for name, value in res["confusion"].items():
            mlflow.log_metric(name, value)

        with tempfile.TemporaryDirectory() as tmp:
            for path in save_artifacts(args, res, tmp):
                mlflow.log_artifact(str(path))

        model = res["model"]
        signature = infer_signature(res["X_train"], model.predict(res["X_train"]))
        mlflow.sklearn.log_model(model, name="modelo", signature=signature,
                                 input_example=res["X_train"].head(3),
                                 skops_trusted_types=["numpy.dtype"])
        run_id = mlflow.active_run().info.run_id
    return run_id


def main(argv=None):
    args = parse_args(argv)
    res = run_training(args)
    run_id = log_to_mlflow(args, res)
    m = res["metrics"]
    print(f"Run {run_id} | acc={m['accuracy']:.4f} f1={m['f1']:.4f} "
          f"recall={m['recall']:.4f} roc_auc={m['roc_auc']:.4f}")


if __name__ == "__main__":
    main()

"""Treinamento do modelo de risco de defasagem da Passos Mágicos.

O protocolo usa indicadores do ano t para prever defasagem no ano t+1.
Treino/seleção: 2022 -> 2023. Teste temporal: 2023 -> 2024.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

RANDOM_STATE = 42
NUMERIC_FEATURES = [
    "idade",
    "anos_programa",
    "iaa",
    "ieg",
    "ips",
    "ida",
    "ipv",
    "ian",
    "defasagem_atual",
    "risco_atual",
    "gap_autoavaliacao_desempenho",
    "media_ida_ieg",
    "min_indicadores",
    "qtd_indicadores_ausentes",
]
CATEGORICAL_FEATURES = ["fase", "genero", "instituicao"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _clean_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().replace({"": pd.NA, "nan": pd.NA})


def load_year(workbook: str | Path, year: int) -> pd.DataFrame:
    """Lê e harmoniza uma aba PEDE sem incorporar colunas futuras."""
    frame = pd.read_excel(workbook, sheet_name=f"PEDE{year}")
    rename = {
        "Fase": "fase",
        "Gênero": "genero",
        "Ano ingresso": "ano_ingresso",
        "Instituição de ensino": "instituicao",
        "IAA": "iaa",
        "IEG": "ieg",
        "IPS": "ips",
        "IDA": "ida",
        "IPV": "ipv",
        "IAN": "ian",
        "Defas": "defasagem",
        "Defasagem": "defasagem",
    }
    frame = frame.rename(columns=rename)
    frame["RA"] = _clean_text(frame["RA"]).str.upper()

    if year == 2022:
        frame["idade"] = pd.to_numeric(frame["Idade 22"], errors="coerce")
    else:
        idade_informada = pd.to_numeric(frame["Idade"], errors="coerce")
        nascimento = pd.to_datetime(frame["Data de Nasc"], errors="coerce")
        idade_derivada = year - nascimento.dt.year
        frame["idade"] = idade_informada.fillna(idade_derivada)

    for column in ["ano_ingresso", "iaa", "ieg", "ips", "ida", "ipv", "ian", "defasagem"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in ["fase", "genero", "instituicao"]:
        frame[column] = _clean_text(frame[column])

    frame["ano"] = year
    frame["risco"] = (frame["defasagem"] < 0).astype("int8")
    if frame["RA"].duplicated().any():
        raise ValueError(f"A aba PEDE{year} contém RA duplicado.")
    return frame


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Cria somente atributos conhecidos no ano corrente."""
    result = frame.copy()
    result["anos_programa"] = (result["ano"] - result["ano_ingresso"]).clip(lower=0)
    result["defasagem_atual"] = result["defasagem"]
    result["risco_atual"] = (result["defasagem"] < 0).astype("int8")
    result["gap_autoavaliacao_desempenho"] = result["iaa"] - result["ida"]
    result["media_ida_ieg"] = result[["ida", "ieg"]].mean(axis=1)
    indicator_columns = ["iaa", "ieg", "ips", "ida", "ipv", "ian"]
    result["min_indicadores"] = result[indicator_columns].min(axis=1)
    result["qtd_indicadores_ausentes"] = result[indicator_columns].isna().sum(axis=1)
    return result


def build_transition(current: pd.DataFrame, following: pd.DataFrame) -> pd.DataFrame:
    """Associa o aluno ao resultado observado no ano seguinte."""
    current = engineer_features(current)
    target = following[["RA", "risco", "defasagem"]].rename(
        columns={"risco": "target_risco_proximo_ano", "defasagem": "defasagem_proximo_ano"}
    )
    transition = current.merge(target, on="RA", how="inner", validate="one_to_one")
    transition["entrou_em_risco"] = (
        (transition["risco_atual"] == 0) & (transition["target_risco_proximo_ano"] == 1)
    ).astype("int8")
    transition["piorou_defasagem"] = (
        transition["defasagem_proximo_ano"] < transition["defasagem_atual"]
    ).astype("int8")
    return transition


def prepare_datasets(workbook: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    years = {year: load_year(workbook, year) for year in (2022, 2023, 2024)}
    train = build_transition(years[2022], years[2023])
    temporal_test = build_transition(years[2023], years[2024])
    all_labeled = pd.concat([train, temporal_test], ignore_index=True)
    return train, temporal_test, all_labeled


def make_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        [("imputer", SimpleImputer(strategy="median", add_indicator=True)), ("scaler", StandardScaler())]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric, NUMERIC_FEATURES), ("categorical", categorical, CATEGORICAL_FEATURES)]
    )


def model_searches(y_train: pd.Series) -> dict[str, tuple[Pipeline, dict]]:
    imbalance = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    models = {
        "regressao_logistica": (
            LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE),
            {"model__C": [0.1, 1.0, 10.0]},
        ),
        "random_forest": (
            RandomForestClassifier(
                n_estimators=500, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
            ),
            {"model__max_depth": [5, None], "model__min_samples_leaf": [3, 8]},
        ),
        "xgboost": (
            XGBClassifier(
                n_estimators=350,
                objective="binary:logistic",
                eval_metric="logloss",
                subsample=0.85,
                colsample_bytree=0.85,
                scale_pos_weight=imbalance,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            {"model__max_depth": [2, 3], "model__learning_rate": [0.03, 0.08]},
        ),
    }
    return {
        name: (Pipeline([("preprocessor", make_preprocessor()), ("model", estimator)]), grid)
        for name, (estimator, grid) in models.items()
    }


def choose_threshold(y_true: pd.Series, probabilities: np.ndarray, beta: float = 2.0) -> float:
    """Seleciona no treino o limiar que maximiza F-beta, priorizando recall."""
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    precision, recall = precision[:-1], recall[:-1]
    beta2 = beta**2
    scores = (1 + beta2) * precision * recall / np.maximum(beta2 * precision + recall, 1e-12)
    return float(thresholds[int(np.nanargmax(scores))])


def classification_metrics(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> dict:
    predicted = (probabilities >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "confusion_matrix": confusion_matrix(y_true, predicted).tolist(),
        "positive_rate": float(np.mean(y_true)),
        "predicted_positive_rate": float(np.mean(predicted)),
        "n": int(len(y_true)),
    }


def train_compare(train: pd.DataFrame, temporal_test: pd.DataFrame) -> tuple[str, Pipeline, float, dict]:
    X_train, y_train = train[MODEL_FEATURES], train["target_risco_proximo_ano"]
    X_test, y_test = temporal_test[MODEL_FEATURES], temporal_test["target_risco_proximo_ano"]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    report: dict[str, dict] = {}
    fitted: dict[str, Pipeline] = {}
    thresholds: dict[str, float] = {}

    for name, (pipeline, parameter_grid) in model_searches(y_train).items():
        search = GridSearchCV(
            pipeline,
            parameter_grid,
            scoring={"precision": "precision", "recall": "recall", "roc_auc": "roc_auc"},
            refit="roc_auc",
            cv=cv,
            n_jobs=-1,
            return_train_score=False,
        )
        search.fit(X_train, y_train)
        best = search.best_estimator_
        oof = cross_val_predict(best, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
        threshold = choose_threshold(y_train, oof, beta=2.0)
        test_probabilities = best.predict_proba(X_test)[:, 1]
        best_index = search.best_index_
        report[name] = {
            "best_params": search.best_params_,
            "cv_precision": float(search.cv_results_["mean_test_precision"][best_index]),
            "cv_recall": float(search.cv_results_["mean_test_recall"][best_index]),
            "cv_roc_auc": float(search.cv_results_["mean_test_roc_auc"][best_index]),
            "oof_threshold_f2": threshold,
            "temporal_test": classification_metrics(y_test, test_probabilities, threshold),
        }
        report[name]["robustness_score"] = float(
            min(report[name]["cv_roc_auc"], report[name]["temporal_test"]["roc_auc"])
        )
        fitted[name] = best
        thresholds[name] = threshold

    # Evita publicar um modelo que pareça bom no treino, mas colapse fora do tempo.
    # O score conservador considera o pior ROC-AUC entre CV e validação temporal.
    selected_name = max(report, key=lambda name: report[name]["robustness_score"])
    return selected_name, fitted[selected_name], thresholds[selected_name], report


def dataset_summary(train: pd.DataFrame, temporal_test: pd.DataFrame) -> dict:
    return {
        "train_2022_to_2023": {
            "n": int(len(train)),
            "positives": int(train["target_risco_proximo_ano"].sum()),
            "positive_rate": float(train["target_risco_proximo_ano"].mean()),
        },
        "test_2023_to_2024": {
            "n": int(len(temporal_test)),
            "positives": int(temporal_test["target_risco_proximo_ano"].sum()),
            "positive_rate": float(temporal_test["target_risco_proximo_ano"].mean()),
        },
        "overlap_ra_train_test": int(len(set(train["RA"]) & set(temporal_test["RA"]))),
    }


def save_outputs(
    output_dir: str | Path,
    selected_name: str,
    evaluated_pipeline: Pipeline,
    threshold: float,
    report: dict,
    train: pd.DataFrame,
    temporal_test: pd.DataFrame,
    all_labeled: pd.DataFrame,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Após a avaliação intocada, reaprende com todos os pares rotulados para uso em 2025.
    final_pipeline = clone(evaluated_pipeline)
    final_pipeline.fit(all_labeled[MODEL_FEATURES], all_labeled["target_risco_proximo_ano"])
    created_at = datetime.now(timezone.utc).isoformat()
    metadata = {
        "model_name": selected_name,
        "target": "target_risco_proximo_ano = 1 quando defasagem no próximo ano < 0",
        "prediction_horizon": "um ano",
        "decision_threshold": float(threshold),
        "features": MODEL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "dataset": dataset_summary(train, temporal_test),
        "model_comparison": report,
        "selection_protocol": (
            "maior min(ROC-AUC da CV 2022->2023, ROC-AUC temporal 2023->2024); "
            "a métrica temporal deve ser interpretada como validação, pois participa da seleção"
        ),
        "evaluated_before_refit": True,
        "final_refit_rows": int(len(all_labeled)),
        "created_at_utc": created_at,
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "joblib": joblib.__version__,
        },
    }
    bundle = {"pipeline": final_pipeline, "threshold": float(threshold), "metadata": metadata}
    model_path = output_dir / "modelo_risco_defasagem.joblib"
    joblib.dump(bundle, model_path)
    (output_dir / "model_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    schema = {
        "required_columns": MODEL_FEATURES,
        "output": {"probability": "float entre 0 e 1", "risk": f"probability >= {threshold:.6f}"},
    }
    (output_dir / "feature_schema.json").write_text(
        json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"model_path": str(model_path), "metadata": metadata}


def run_training(workbook: str | Path, output_dir: str | Path) -> dict:
    train, temporal_test, all_labeled = prepare_datasets(workbook)
    selected_name, evaluated_pipeline, threshold, report = train_compare(train, temporal_test)
    return save_outputs(
        output_dir,
        selected_name,
        evaluated_pipeline,
        threshold,
        report,
        train,
        temporal_test,
        all_labeled,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Caminho para BASE DE DADOS PEDE 2024 - DATATHON.xlsx")
    parser.add_argument("--output", default="models", help="Diretório dos artefatos")
    args = parser.parse_args()
    result = run_training(args.data, args.output)
    print(json.dumps(result["metadata"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

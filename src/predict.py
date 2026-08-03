"""Funções mínimas para consumir o modelo no Streamlit."""

from pathlib import Path

import joblib
import pandas as pd


def load_model(path: str | Path = "models/modelo_risco_defasagem.joblib") -> dict:
    bundle = joblib.load(path)
    expected = {"pipeline", "threshold", "metadata"}
    if not expected.issubset(bundle):
        raise ValueError(f"Artefato inválido; chaves esperadas: {sorted(expected)}")
    return bundle


def predict_risk(bundle: dict, students: pd.DataFrame) -> pd.DataFrame:
    features = bundle["metadata"]["features"]
    missing = sorted(set(features) - set(students.columns))
    if missing:
        raise ValueError(f"Colunas ausentes: {missing}")
    probabilities = bundle["pipeline"].predict_proba(students[features])[:, 1]
    result = students.copy()
    result["probabilidade_risco"] = probabilities
    result["risco_previsto"] = probabilities >= bundle["threshold"]
    return result

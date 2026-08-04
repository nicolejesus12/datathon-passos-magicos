from pathlib import Path
import joblib
import pandas as pd
import streamlit as st


@st.cache_resource
def load_model(path: str | Path = "models/modelo_risco_defasagem.joblib") -> dict:
    """Carrega o artefato do modelo com cache do Streamlit."""
    bundle = joblib.load(path)
    expected = {"pipeline", "threshold", "metadata"}
    if not expected.issubset(bundle):
        raise ValueError(f"Artefato inválido; chaves esperadas: {sorted(expected)}")
    return bundle


def predict_risk(bundle: dict, students: pd.DataFrame) -> pd.DataFrame:
    """Prepara os dados do aluno e realiza a predição usando o pipeline treinado."""
    df = students.copy()
    
    # 1. Nomes em minúsculo
    df.columns = [c.lower() for c in df.columns]

    # 2. Obtém a lista exata de colunas que o modelo espera
    features = bundle["metadata"]["features"]

    # 3. Preenche colunas calculadas/derivadas se existirem no modelo
    if 'media_ida_ieg' in features and 'media_ida_ieg' not in df.columns:
        ida = df.get('ida', 8.0)
        ieg = df.get('ieg', 8.0)
        df['media_ida_ieg'] = (ida + ieg) / 2.0

    if 'gap_autoavaliacao_desempenho' in features and 'gap_autoavaliacao_desempenho' not in df.columns:
        iaa = df.get('iaa', 8.0)
        ida = df.get('ida', 8.0)
        df['gap_autoavaliacao_desempenho'] = iaa - ida

    # 4. Dicionário de valores padrão realistas/neutros (em vez de 0)
    defaults_medianos = {
        'anos_programa': 2,
        'fase': 3,
        'defasagem_atual': 0,
        'idade': 12,
        'ipv': 8.0,                     # Nota neutra/alta para não forçar o risco
        'min_indicadores': 7.0,          # Média saudável
        'qtd_indicadores_ausentes': 0,   # Sem dados ausentes
        'risco_atual': 0,                # Sem risco histórico
        'ian': 8.0,
        'ida': 8.0,
        'ieg': 8.0,
        'iaa': 8.0,
        'ips': 8.0,
        'ipp': 8.0
    }

    # 5. Preenche colunas ausentes com valores padrão sensatos
    for col in features:
        if col not in df.columns:
            if col in ['genero', 'instituicao']:
                df[col] = "Não Informado"
            else:
                df[col] = defaults_medianos.get(col, 5.0)

    # 6. Ordena o DataFrame na ordem exata do modelo
    df_features = df[features]

    # 7. Executa a predição
    probabilities = bundle["pipeline"].predict_proba(df_features)[:, 1]
    
    result = students.copy()
    result["probabilidade_risco"] = probabilities
    result["risco_previsto"] = probabilities >= bundle.get("threshold", 0.5)
    
    return result
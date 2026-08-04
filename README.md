# Datathon Passos Mágicos — PosTech
Solução preditiva e analítica de dados desenvolvida para a Associação Passos Mágicos.

## Estrutura do repositório;

- `.streamlit/`: Configurações de tema do Streamlit.
- `data/`: Conjunto de dados e bases de análise.
- `models/`: Modelos preditivos treinados (`.pkl`).
- `notebooks/`: Análises exploratórias (EDA) e desenvolvimento do modelo de ML.
- `src/`: Scripts auxiliares em Python.
- `app.py`: Aplicação web interativa no Streamlit.
- `requirements.txt`: Dependências do projeto.

# Associação Passos Mágicos — Portal de Decisão e Predição de Risco

> **Datathon — PosTech Data Analytics (FIAP)**  
> Plataforma desenvolvida para auxílio a gestores e equipe pedagógica na identificação precoce de risco de defasagem acadêmica em alunos do programa PEDE.

---

## Integrantes do Grupo

* **Ana Monteiro** — EDA & Analytics
* **Guilherme Roxo** — Storytelling & Dashboard
* **Gustavo Martins** — Machine Learning & Modelagem
* **Nicole Jesus** — Engenharia de Software & Streamlit

---

## Objetivo do Projeto

Transformar os dados históricos da Associação Passos Mágicos (2022–2024) em inteligência acionável por meio de:
1. **Análise Exploratória e Diagnóstico:** Acompanhamento da evolução de KPIs (IAN, IDA, IEG, etc.).
2. **Modelo Preditivo (Machine Learning):** Classificação automática do risco de defasagem para rápida intervenção psicopedagógica.
3. **Interface de Decisão:** Aplicação interativa desenvolvida em Streamlit para uso direto pela equipe do programa.

---

## Estrutura do Repositório

```text
datathon-passos-magicos/
├── app.py                   # Aplicação principal do Streamlit
├── requirements.txt         # Dependências do projeto
├── README.md                # Documentação técnica do projeto
├── models/
│   └── modelo_risco_defasagem.joblib   # Artefato do modelo treinado
├── reports/
│   └── Passos_Magicos_Storytelling.pdf # Apresentação em PDF
└── src/
    ├── __init__.py          # Identificador do pacote Python
    ├── predict.py           # Módulo de carregamento e inferência do modelo
    └── logo-passos-magicos.png
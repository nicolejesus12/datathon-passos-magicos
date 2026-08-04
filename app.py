import streamlit as st
import pandas as pd
import numpy as np
import os

# Configuração da página
st.set_page_config(
    page_title="Passos Mágicos — Portal de Decisão",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilização do tema e fontes
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    html, body, [class*="css"], .stMarkdown, p, button, input {
        font-family: 'Poppins', sans-serif !important;
    }
    h1, h2, h3 {
        color: #004085;
        font-weight: 700;
    }
    </style>
""", unsafe_allow_html=True)

# Caminho da imagem
LOGO_PATH = os.path.join("src", "logo-passos-magicos.png")

# Barra Lateral (Sidebar)
with st.sidebar:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=200)
    else:
        st.title("🎓 Passos Mágicos")
        
    st.markdown("---")
    st.title("Navegação")
    st.info("Projeto Datathon — PosTech Data Analytics")
    st.markdown("---")
    st.markdown("**Integrantes do Grupo:**")
    st.caption("• Integrante 1 (EDA)\n• Integrante 2 (Storytelling)\n• Integrante 3 (Machine Learning)\n• Integrante 4 (Engenharia & Streamlit)")

# Cabeçalho Principal
st.title("Associação Passos Mágicos")
st.subheader("Plataforma de Analytics e Predição de Risco Acadêmico")
st.markdown("---")

# Estrutura em Abas
aba1, aba2, aba3 = st.tabs([
    "Visão Geral & Contexto", 
    "Dashboard de Indicadores", 
    "Modelo Preditivo (Risco)"
])

# ABA 1: CONTEXTO DO PROJETO
with aba1:
    st.header("Sobre a Passos Mágicos")
    st.write(
        "A Associação Passos Mágicos tem uma trajetória de 35 anos atuando na transformação "
        "da vida de crianças e jovens de baixa renda por meio da educação de qualidade, auxílio "
        "psicológico/psicopedagógico e ampliação da visão de mundo."
    )
    col_a, col_b = st.columns(2)
    with col_a:
        st.info("**Objetivo:** Identificar precocemente alunos com risco de defasagem acadêmica para intervenção rápida.")
    with col_b:
        st.success("**Público-Alvo:** Equipe pedagógica, psicólogos e gestores da Passos Mágicos.")

# ABA 2: DASHBOARD DE INDICADORES (Integração dos Achados do Tópico 1)
with aba2:
    st.header("Evolução dos Indicadores (2022 – 2024) 📊")
    st.write("Resultados da Análise Exploratória de Dados consolidados do programa PEDE.")

    # Cards com KPIs Principais
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric(label="Alunos sem Defasagem (IAN = 10)", value="53.81%", delta="+23.69% vs 2022")
    with col_m2:
        st.metric(label="Defasagem Severa (IAN = 2.5)", value="0.26%", delta="-3.00% vs 2022")
    with col_m3:
        st.metric(label="Desempenho Acadêmico Médio (IDA)", value="6.35", delta="+0.26 vs 2022")

    st.markdown("---")
    
    with st.container(border=True):
        st.subheader("Resumo das Análises Exploratórias")
        st.markdown("""
        * **Adequação do Nível (IAN):** A proporção de alunos no nível máximo de adequação ($IAN = 10,0$) subiu de **30,12% (2022)** para **53,81% (2024)**, enquanto a defasagem severa ($IAN = 2,5$) foi reduzida de **3,26%** para apenas **0,26%**.
        * **Desempenho Acadêmico (IDA):** A média geral do desempenho evoluiu de **6,09 (2022)** para **6,66 (2023)** e estabilizou em **6,35 (2024)**, demonstrando avanço consistente ao longo do ciclo.
        """)

# ABA 3: SIMULAÇÃO PREDITIVA
with aba3:
    st.header("Simulação Preditiva de Risco de Defasagem")
    st.write("Insira os indicadores do aluno para simular a probabilidade de risco:")

    col1, col2 = st.columns(2)
    with col1:
        ian = st.slider("IAN (Adequação do Nível)", 0.0, 10.0, 5.0)
        ida = st.slider("IDA (Desempenho Acadêmico)", 0.0, 10.0, 5.0)
        ieg = st.slider("IEG (Engajamento)", 0.0, 10.0, 5.0)

    with col2:
        iaa = st.slider("IAA (Autoavaliação)", 0.0, 10.0, 5.0)
        ips = st.slider("IPS (Aspectos Psicossociais)", 0.0, 10.0, 5.0)
        ipp = st.slider("IPP (Aspectos Psicopedagógicos)", 0.0, 10.0, 5.0)

    st.markdown("---")
    
    if st.button("Calcular Risco do Aluno", type="primary"):
        st.subheader("Resultado da Análise:")
        media_score = (ian + ida + ieg + ips) / 4
        if media_score < 5.0:
            st.error(f"**Alto Risco de Defasagem** (Score Médio: {media_score:.1f})")
            st.write("Recomendação: Encaminhar para acompanhamento psicopedagógico prioritário.")
        else:
            st.success(f"**Baixo Risco de Defasagem** (Score Médio: {media_score:.1f})")
            st.write("Recomendação: Manter acompanhamento regular de engajamento.")
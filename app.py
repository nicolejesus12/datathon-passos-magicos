import streamlit as st
import pandas as pd
import numpy as np

# Configuração da página
st.set_page_config(
    page_title="Passos Mágicos — Portal de Decisão",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cabeçalho Principal
st.title("Associação Passos Mágicos")
st.subheader("Plataforma de Analytics e Predição de Risco Acadêmico")
st.markdown("---")

# Barra Lateral (Sidebar)
st.sidebar.image("src/logo-passos-magicos.png", width=200)
st.sidebar.title("Navegação")
st.sidebar.info("Projeto Datathon — PosTech Data Analytics")

# Estrutura em Abas
aba1, aba2, aba3 = st.tabs([
    "Visão Geral & Contexto", 
    "Dashboard de Indicadores", 
    "Modelo Preditivo (Risco)"
])

# Contexto do Projeto
with aba1:
    st.header("Sobre a Passos Mágicos")
    st.write(
        "A Associação Passos Mágicos tem uma trajetória de 35 anos atuando na transformação "
        "da vida de crianças e jovens de baixa renda por meio da educação de qualidade, auxílio "
        "psicológico/psicopedagógico e ampliação da visão de mundo."
    )
    st.info("**Objetivo desta Aplicação:** Disponibilizar uma ferramenta prática para identificar precocemente alunos com risco de defasagem acadêmica.")

# Dashboard
with aba2:
    st.header("Análise Exploratória dos Indicadores")
    st.write("Esta seção apresentará a evolução dos indicadores (IAN, IDA, IEG, IPS, IPP, IPV e INDE).")
    st.warning("Os gráficos e análises serão integrados conforme a conclusão das etapas do grupo.")

# Predição de risco
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
    
    if st.button("Calcular Risco do Aluno"):
        # Lógica temporária até o modelo real (.pkl) ser carregado
        st.subheader("Resultado da Análise (Simulação):")
        
        # Exemplo simulado simples
        media_score = (ian + ida + ieg + ips) / 4
        if media_score < 5.0:
            st.error(f"**Alto Risco de Defasagem** (Score Médio: {media_score:.1f})")
            st.write("Recomendação: Encaminhar para acompanhamento psicopedagógico prioritário.")
        else:
            st.success(f"**Baixo Risco de Defasagem** (Score Médio: {media_score:.1f})")
            st.write("Recomendação: Manter acompanhamento regular de engajamento.")
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Análise ML · Afastamentos",
    page_icon="⬜",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    .stTabs [data-baseweb="tab"] { font-size: 0.85rem; font-weight: 600; letter-spacing: 0.04em; }
    div[data-testid="metric-container"] {
        border: 1px solid rgba(128,128,128,0.2);
        border-radius: 8px;
        padding: 12px 16px;
    }
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
CID_MAP = {
    'A': 'Infecciosas', 'B': 'Infecciosas',
    'C': 'Neoplasias',  'D': 'Neoplasias',
    'E': 'Metabolicas',
    'F': 'Saude Mental',
    'G': 'Neurologicas',
    'H': 'Olhos/Ouvidos',
    'I': 'Cardiovascular',
    'J': 'Respiratorias',
    'K': 'Digestivas',
    'L': 'Pele',
    'M': 'Musculoesqueletico',
    'N': 'Genito-urinario',
    'R': 'Sintomas Inesp.',
    'S': 'Lesoes/Traumas', 'T': 'Lesoes/Traumas',
    'Z': 'Fatores Saude',
}

def categorize(cid):
    c = str(cid).strip().upper()
    return CID_MAP.get(c[0], 'Outros')

import os

# ── Arquivos da base de dados ─────────────────────────────────────────────────
# Todos os CSVs devem estar na mesma pasta do app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ARQUIVOS = [
    # 2025
    ('BASE DE DADOS DEZEMBRO 2025.csv',  'DEZ/25'),
    # 2026
    ('BASE DE DADOS JANEIRO 2026.csv',   'JAN/26'),
    ('BASE DE DADOS FEVEREIRO 2026.csv', 'FEV/26'),
    # ('BASE DE DADOS MARCO 2026.csv',     'MAR/26'),
    # ('BASE DE DADOS ABRIL 2026.csv',     'ABR/26'),
    # ('BASE DE DADOS MAIO 2026.csv',      'MAI/26'),
    # ('BASE DE DADOS JUNHO 2026.csv',     'JUN/26'),
    # ('BASE DE DADOS JULHO 2026.csv',     'JUL/26'),
    # ('BASE DE DADOS AGOSTO 2026.csv',    'AGO/26'),
    # ('BASE DE DADOS SETEMBRO 2026.csv',  'SET/26'),
    # ('BASE DE DADOS OUTUBRO 2026.csv',   'OUT/26'),
    # ('BASE DE DADOS NOVEMBRO 2026.csv',  'NOV/26'),
    # ('BASE DE DADOS DEZEMBRO 2026.csv',  'DEZ/26'),
]

@st.cache_data
def load_data():
    dfs = []
    for arquivo, label in ARQUIVOS:
        caminho = os.path.join(BASE_DIR, arquivo)
        if os.path.exists(caminho):
            df = pd.read_csv(caminho, dtype={'MAT': str})
            df['MAT'] = df['MAT'].str.zfill(6)
            df['MES'] = label
            dfs.append(df)
        else:
            st.warning(f"Arquivo não encontrado: {arquivo}")
    if not dfs:
        st.error("Nenhum arquivo CSV encontrado na pasta do app.")
        st.stop()
    df = pd.concat(dfs, ignore_index=True)
    df['CATEGORIA'] = df['CID'].apply(categorize)
    df['DATA'] = pd.to_datetime(df['DATA'], dayfirst=True, errors='coerce')
    # Remove datas fora do intervalo esperado da base (DEZ/2025 a DEZ/2026)
    df = df[(df['DATA'] >= '2025-12-01') & (df['DATA'] <= '2026-12-31') | df['DATA'].isna()]
    return df

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Análise de Afastamentos")
    st.markdown("---")
    st.markdown("### Parâmetros ML")
    anomaly_threshold = st.slider("Limiar de anomalia (desvios-padrão)", 1.5, 4.0, 2.5, 0.1)
    n_epochs = st.slider("Epocas do Autoencoder", 10, 100, 30, 5)
    run_tf = st.checkbox("Usar Keras Autoencoder", value=True)
    st.markdown("---")
    st.caption("Métodos: Autoencoder · IQR · Clustering · Série Temporal")

    col1, col2, col3 = st.sidebar.columns([1, 2, 1])
    with col2:
        if os.path.exists("lalo.png"):
            st.image("lalo.png", use_container_width=True)

st.sidebar.markdown("""
<div style='text-align: center;'>
    <p style='color: #64748b; font-size: 0.8rem; margin: 0;'>Da planilha ao modelo de IA.</p>
    <p style='color: #64748b; font-size: 0.8rem; margin: 0;'>Coleta, tratamento e análise de dados com métodos de Machine Learning em Python.</p>
    <p style='color: #64748b; font-size: 0.8rem; margin: 0;'>Python · Numpy · Pandas · Streamlit · Keras · Scilit-Learn</p>
    <p style='color: #1e3a5f; font-weight: 600; margin: 6px 0 2px 0;'>Jorge Eduardo de Araujo Oliveira</p>
    <p style='color: #64748b; font-size: 0.8rem; margin: 0;'>Tecnólogo em Análise e Desenvolvimento de Sistemas</p>
</div>
""", unsafe_allow_html=True)

df = load_data()

# ── KPIs ──────────────────────────────────────────────────────────────────────
meses_carregados = ' · '.join([lbl for arq, lbl in ARQUIVOS if os.path.exists(os.path.join(BASE_DIR, arq))])
st.title("Análise ML · Afastamentos")
st.caption(f"Base consolidada: {len(df)} registros · {meses_carregados}")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total de Casos", len(df))
with col2:
    st.metric("Dias Perdidos", int(df['DIAS'].sum()))
with col3:
    st.metric("Média por Caso", f"{df['DIAS'].mean():.1f} dias")
with col4:
    Q1, Q3 = df['DIAS'].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    n_outliers = (df['DIAS'] > Q3 + 1.5 * IQR).sum()
    st.metric("Anomalias Detectadas (IQR)", n_outliers)

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "Visão Geral",
    "Autoencoder — Keras",
    "Reincidência",
    "Análise Temporal"
])

# ════════════════════════════════════════════════════════════════════
# TAB 1 — Visão Geral
# ════════════════════════════════════════════════════════════════════
with tab1:
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Casos por Categoria CID-10")
        cat = df.groupby('CATEGORIA').agg(casos=('DIAS','count'), media=('DIAS','mean')).reset_index()
        fig = px.bar(cat.sort_values('casos'), x='casos', y='CATEGORIA', orientation='h',
                     color='media', color_continuous_scale='teal',
                     labels={'casos': 'Casos', 'CATEGORIA': '', 'media': 'Média dias'})
        fig.update_layout(height=400, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, width="stretch")

    with col_b:
        st.subheader("Clustering por Severidade")
        df['CLUSTER'] = pd.cut(df['DIAS'], bins=[0,2,7,999],
                               labels=['Leve (1-2d)', 'Moderado (3-7d)', 'Grave (>7d)'])
        cluster_stats = df.groupby('CLUSTER', observed=True).agg(
            casos=('DIAS','count'),
            total_dias=('DIAS','sum'),
            media=('DIAS','mean')
        ).reset_index()
        colors = ['#2dd4bf', '#fbbf24', '#f87171']
        fig2 = go.Figure()
        for i, row in cluster_stats.iterrows():
            fig2.add_trace(go.Bar(
                name=str(row['CLUSTER']),
                x=['Casos', 'Dias Totais'],
                y=[row['casos'], row['total_dias']],
                marker_color=colors[i],
                text=[str(row['casos']), str(row['total_dias'])],
                textposition='outside',
            ))
        fig2.update_layout(
            barmode='group',
            height=400,
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
            yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.15)'),
            xaxis=dict(showgrid=False),
        )
        st.plotly_chart(fig2, width="stretch")

    st.subheader("Top 15 CIDs — Casos vs Dias Totais")
    top_cids = df.groupby('CID').agg(casos=('DIAS','count'), total_dias=('DIAS','sum')).reset_index()
    top_cids = top_cids.sort_values('casos', ascending=False).head(15)
    fig3 = make_subplots(specs=[[{"secondary_y": True}]])
    fig3.add_trace(go.Bar(x=top_cids['CID'], y=top_cids['casos'],
                          name='Casos', marker_color='#2dd4bf'), secondary_y=False)
    fig3.add_trace(go.Scatter(x=top_cids['CID'], y=top_cids['total_dias'],
                              name='Dias Totais', mode='lines+markers',
                              line=dict(color='#f97316', width=2)), secondary_y=True)
    fig3.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig3, width="stretch")

    st.subheader("Impacto por Categoria — Dias Médios de Afastamento")
    media_cat = df.groupby('CATEGORIA')['DIAS'].mean().reset_index().sort_values('DIAS', ascending=False)
    fig4 = px.bar(media_cat, x='CATEGORIA', y='DIAS',
                  color='DIAS', color_continuous_scale='RdYlGn_r',
                  labels={'DIAS': 'Média de Dias', 'CATEGORIA': 'Categoria'})
    fig4.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
    st.plotly_chart(fig4, width="stretch")

# ════════════════════════════════════════════════════════════════════
# TAB 2 — Autoencoder Keras
# ════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Detecção de Anomalias com Autoencoder — Keras")
    st.markdown(
        "O autoencoder aprende a reconstruir padrões normais de afastamento. "
        "Registros com alto erro de reconstrução são sinalizados como anomalias — "
        "aprendizado **não-supervisionado**, sem necessidade de rótulos."
    )

    if run_tf:
        try:
            with st.spinner("Treinando autoencoder..."):
                feat = df.copy()
                feat['CAT_ENC'] = LabelEncoder().fit_transform(feat['CATEGORIA'])
                feat['MES_ENC'] = LabelEncoder().fit_transform(feat['MES'])

                X = feat[['DIAS', 'CAT_ENC', 'MES_ENC']].values.astype(np.float32)
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                ae = MLPRegressor(
                    hidden_layer_sizes=(8, 4, 8),
                    activation='relu',
                    max_iter=n_epochs * 10,
                    random_state=42,
                    verbose=False
                )
                ae.fit(X_scaled, X_scaled)

                X_pred = ae.predict(X_scaled)
                mse = np.mean(np.power(X_scaled - X_pred, 2), axis=1)
                threshold = np.mean(mse) + anomaly_threshold * np.std(mse)

                df['RECON_ERROR'] = mse
                df['ANOMALIA'] = mse > threshold

            col_x, col_y = st.columns(2)

            with col_x:
                st.subheader("Loss durante treinamento")
                loss_df = pd.DataFrame({
                    'Iteração': range(1, len(ae.loss_curve_) + 1),
                    'MSE Loss': ae.loss_curve_,
                })
                fig_loss = px.line(loss_df, x='Iteração', y='MSE Loss',
                                   color_discrete_sequence=['#2dd4bf'])
                fig_loss.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig_loss, width="stretch")

            with col_y:
                st.subheader("Erro de Reconstrução por Registro")
                fig_err = px.scatter(df, x=df.index, y='RECON_ERROR',
                                     color='ANOMALIA',
                                     color_discrete_map={True:'#f87171', False:'#2dd4bf'},
                                     hover_data=['CID','MAT','DIAS'],
                                     labels={'RECON_ERROR': 'Erro de Reconstrução', 'index': 'Registro'})
                fig_err.add_hline(y=threshold, line_dash='dash',
                                  line_color='#fbbf24',
                                  annotation_text=f"Limiar ({anomaly_threshold} sigma)")
                fig_err.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig_err, width="stretch")

            n_an = df['ANOMALIA'].sum()
            st.success(
                f"Autoencoder detectou {n_an} anomalias ({n_an/len(df)*100:.1f}% da base) "
                f"com limiar de {anomaly_threshold} desvios-padrão."
            )

            st.subheader("Registros Anômalos")
            anomalos = df[df['ANOMALIA']].sort_values('RECON_ERROR', ascending=False)
            st.dataframe(
                anomalos[['CID','MAT','DATA','DIAS','MES','CATEGORIA','RECON_ERROR']]
                .rename(columns={'RECON_ERROR': 'Erro Reconstr.'})
                .style.format({'Erro Reconstr.': '{:.4f}'}),
                width="stretch"
            )

        except Exception as e:
            st.error(f"Erro: {e}")

    else:
        st.info("Ative o Keras Autoencoder na sidebar para executar o modelo.")
        st.subheader("Anomalias via IQR (método estatístico clássico)")
        Q1, Q3 = df['DIAS'].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        outliers = df[df['DIAS'] > Q3 + 1.5 * IQR].sort_values('DIAS', ascending=False)
        st.dataframe(outliers[['CID','MAT','DATA','DIAS','MES','CATEGORIA']], width="stretch")

# ════════════════════════════════════════════════════════════════════
# TAB 3 — Reincidência
# ════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Análise de Reincidência por Matrícula")

    reinc = df.groupby('MAT').agg(
        ocorrencias=('CID','count'),
        total_dias=('DIAS','sum'),
        media_dias=('DIAS','mean'),
        cids_unicos=('CID', 'nunique'),
        meses=('MES', lambda x: ', '.join(sorted(set(x))))
    ).reset_index().sort_values('ocorrencias', ascending=False)

    reinc['Risco'] = reinc['ocorrencias'].apply(
        lambda x: 'Alto' if x >= 7 else ('Médio' if x >= 4 else 'Baixo')
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Matrículas Únicas", reinc['MAT'].nunique())
    with col2:
        st.metric("Com 3 ou mais ocorrências", (reinc['ocorrencias'] >= 3).sum())
    with col3:
        st.metric("Risco Alto (7+ ocorrências)", (reinc['ocorrencias'] >= 7).sum())

    st.subheader("Top 20 Matrículas — Dias Totais vs Ocorrências")
    reinc_plot = reinc[reinc['ocorrencias'] > 1].head(20).sort_values('total_dias').copy()
    reinc_plot['MAT'] = reinc_plot['MAT'].astype(str)
    fig_reinc = px.bar(
        reinc_plot,
        x='total_dias', y='MAT', color='ocorrencias',
        color_continuous_scale='RdYlGn_r', orientation='h',
        text='total_dias',
        labels={'total_dias': 'Dias Totais', 'MAT': 'Matrícula', 'ocorrencias': 'Ocorrências'},
        category_orders={'MAT': reinc_plot['MAT'].tolist()}
    )
    fig_reinc.update_layout(
        height=520,
        margin=dict(l=120, r=20, t=10, b=40),
        yaxis=dict(type='category', tickfont=dict(size=12), automargin=True),
        xaxis=dict(title='Dias Totais'),
    )
    fig_reinc.update_traces(textposition='outside')
    st.plotly_chart(fig_reinc, width="stretch")

    st.subheader("Tabela Detalhada")
    st.dataframe(
        reinc[reinc['ocorrencias'] > 1][['MAT','ocorrencias','total_dias','media_dias','cids_unicos','meses','Risco']],
        width="stretch"
    )

# ════════════════════════════════════════════════════════════════════
# TAB 4 — Temporal
# ════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Análise Temporal de Afastamentos")

    mensal = df.groupby('MES').agg(
        casos=('DIAS','count'),
        total_dias=('DIAS','sum'),
        media=('DIAS','mean')
    ).reset_index()
    ordem = ['DEZ/25','JAN/26','FEV/26']
    mensal['MES'] = pd.Categorical(mensal['MES'], categories=ordem, ordered=True)
    mensal = mensal.sort_values('MES')

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Casos por Mês")
        fig_m = px.bar(mensal, x='MES', y='casos', color='media',
                       color_continuous_scale='teal',
                       labels={'casos': 'Casos', 'MES': 'Mês', 'media': 'Média dias'})
        fig_m.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_m, width="stretch")

    with col_b:
        st.subheader("Dias Totais e Média — Evolução")
        fig_t = px.line(mensal, x='MES', y=['total_dias','media'],
                        markers=True,
                        color_discrete_sequence=['#7c3aed','#f97316'])
        fig_t.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_t, width="stretch")

    st.subheader("Distribuição de Dias por Mês")
    fig_box = px.box(df, x='MES', y='DIAS', color='MES',
                     color_discrete_sequence=['#2dd4bf','#fbbf24','#f97316'],
                     category_orders={'MES': ordem},
                     points='outliers',
                     labels={'DIAS': 'Dias de Afastamento', 'MES': 'Mês'})
    fig_box.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
    st.plotly_chart(fig_box, width="stretch")

    if df['DATA'].notna().sum() > 10:
        st.subheader("Série Diária — Dias de Afastamento")
        diario = df.dropna(subset=['DATA']).groupby('DATA').agg(
            casos=('DIAS','count'), dias=('DIAS','sum')
        ).reset_index()
        data_min = diario['DATA'].min()
        data_max = diario['DATA'].max()
        # Remove datas espúrias: mantém só o intervalo real da base
        meses_validos = df['MES'].unique()
        diario = diario[(diario['DATA'] >= data_min) & (diario['DATA'] <= data_max)]
        fig_d = px.area(diario, x='DATA', y='dias',
                        labels={'DATA': 'Data', 'dias': 'Total de Dias Afastados'},
                        color_discrete_sequence=['#2dd4bf'])
        fig_d.update_layout(
            height=340,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(range=[data_min, data_max])
        )
        st.plotly_chart(fig_d, width="stretch")

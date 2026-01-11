import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import warnings
from statsmodels.tsa.stattools import adfuller
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import plotly.graph_objects as go
import plotly.express as px


# Configurações iniciais
warnings.filterwarnings("ignore")
st.set_page_config(page_title="Ibovespa Predictor Pro", layout="wide")

st.title("Painel Interativo de Previsões de Ativos Financeiros com XGBoost")

st.markdown("Insira os dados e visualize previsões do mercardo de ações de forma simples e interativa.")


# --- FUNÇÃO PARA CARREGAR O MODELO E METADADOS ---
@st.cache_resource
def load_model():
    model = joblib.load('modelo_xgb_financeiro.joblib')
    
    # Extrair os melhores parâmetros e a instância do classificador
    try:
        # Se veio de um RandomizedSearchCV/GridSearchCV
        params = model.best_params_
        clf_instance = model.best_estimator_.named_steps['clf']
        pipeline_obj = model.best_estimator_
    except:
        # Se for o Pipeline direto
        params = model.get_params()
        clf_instance = model.named_steps['clf']
        pipeline_obj = model
        
    return pipeline_obj, params, clf_instance

# --- FUNÇÃO DE ENGENHARIA DE FEATURES ---
def preparar_dados(ticker_simbolo):
    df = yf.download(ticker_simbolo, period="2y", interval="1d")
    if df.empty: return None, None
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)

    df_calc = pd.DataFrame(index=df.index)
    df_calc["Close"] = df["Close"]
    delta = df_calc["Close"].diff()
    df_calc["Delta"] = delta
    df_calc["Return"] = df_calc["Close"].pct_change().shift(1)
    
    for i in range(1, 8):
        df_calc[f"Delta_lag{i}"] = delta.shift(i)
    
    df_calc['MA5'] = delta.rolling(window=5).mean()
    df_calc['MA22'] = delta.rolling(window=22).mean()
    df_calc['MA66'] = delta.rolling(window=66).mean()
    df_calc['MA132'] = delta.rolling(window=132).mean()
    df_calc['MA252'] = delta.rolling(window=252).mean()
    df_calc["Volatilidade"] = delta.rolling(window=5).std()
    
    predictors = [
        'Delta', 'Return', 'Delta_lag1', 'Delta_lag2', 'Delta_lag3', 
        'Delta_lag4', 'Delta_lag5', 'Delta_lag6', 'Delta_lag7', 
        'MA5', 'MA22', 'MA66', 'MA132', 'MA252', 'Volatilidade'
    ]
    
    input_data = df_calc[predictors].dropna().tail(1)
    return input_data, df

# --- CARREGAR MODELO ---
model_pipeline, best_params, clf_instance = load_model()

# --- INTERFACE ---
st.markdown(
    "<p style='font-size:20px; font-weight:500; color:#2563eb;'>"
    "Modelo de séries temporais treinado com XGBoost para previsão de ativos financeiros em tempo real."
    "</p>",
    unsafe_allow_html=True
)


st.sidebar.header("Menu de Navegação")
ticker = st.sidebar.text_input("Digite o código do Ativo:", value="^BVSP")
st.sidebar.info("Exemplos: ^BVSP (Ibovespa), PETR4.SA, AAPL, BTC-USD")

st.sidebar.divider()

pagina = st.sidebar.radio("Visualização e Monitoramento do Modelo Treinado:", ["📊 Performance do Modelo","⚙️ Parâmetros do Modelo (XGBoost)", "🧠 Importância das Features"])


st.sidebar.divider()

import streamlit as st

# Injeta CSS para centralizar o texto dentro de alertas na sidebar
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] .stAlert {
        text-align: center;
        display: flex;
        justify-content: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.text("Equipe de desenvolvimento:")

st.sidebar.info("""

                  
Eduardo Jorge\n
Erikson Machado\n
Mariangela da Silva\n 
Marcos Aurélio\n
Moacir Carlos 
""")


# Agora o código abaixo executa sempre que o 'ticker' mudar
if ticker:
    with st.spinner(f'Analisando {ticker}...'):
        # 1. Busca os dados (Importante: use cache para não travar o app a cada tecla)
        input_data, df_full = preparar_dados(ticker)
        
        if input_data is not None and not input_data.empty:
            # 2. Realiza a Previsão
            pred = model_pipeline.predict(input_data)[0]
            prob = model_pipeline.predict_proba(input_data)[0]

            # 3. Exibe as Métricas
            col1, col2, col3 = st.columns(3)
            col1.metric("Preço Atual", f"{df_full['Close'].iloc[-1]:,.2f}")
             # Probabilidade da classe positiva (alta)
            prob_alta = prob[1]

            # Definição do sinal (3 estados)
            if prob_alta >= 0.60:
                sinal = "📈 COMPRAR"
            elif prob_alta >= 0.45:
                sinal = "⏳ AGUARDAR"
            else:
                sinal = "📉 EVITAR"

            # Métrica
            col2.metric("Sinal Sugerido", sinal)


            # Probabilidade da classe positiva (alta)
            prob_alta = prob[1]

            if prob_alta >= 0.60:
                st.success(
                    f"### 📈 COMPRAR\n"
                    f"Probabilidade de alta: **{prob_alta*100:.1f}%**"
                )

            elif prob_alta >= 0.45:
                st.info(
                    f"### ⏳ AGUARDAR\n"
                    f"Probabilidade de alta: **{prob_alta*100:.1f}%**"
                )

            else:
                st.warning(
                    f"### 📉 EVITAR\n"
                    f"Probabilidade de alta: **{prob_alta*100:.1f}%**"
                )

            # 5. Gráfico de Candlestick
            st.subheader(f"Visão de Mercado - {ticker}")
            fig_candle = go.Figure(data=[go.Candlestick(
                x=df_full.tail(60).index,
                open=df_full.tail(60)['Open'], 
                high=df_full.tail(60)['High'],
                low=df_full.tail(60)['Low'], 
                close=df_full.tail(60)['Close'],
                name="Candlesticks")])
            
            fig_candle.update_layout(
                template="plotly_dark", 
                height=500, 
                xaxis_rangeslider_visible=False
            )
            st.plotly_chart(fig_candle, use_container_width=True)
        else:
            st.error("Ticker não encontrado ou dados insuficientes.")

# --- CARREGAR MODELO ---
model_pipeline, best_params, clf_instance = load_model()

# --- FUNÇÃO DE ENGENHARIA DE FEATURES (Reutilizável) ---
def preparar_dados_completos(ticker, period="5y"):
    df = yf.download(ticker, period=period)
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    
    df_calc = pd.DataFrame(index=df.index)
    df_calc["Close"] = df["Close"]
    delta = df_calc["Close"].diff()
    df_calc["Delta"] = delta
    df_calc["Return"] = df_calc["Close"].pct_change().shift(1)
    
    for i in range(1, 8):
        df_calc[f"Delta_lag{i}"] = delta.shift(i)
    
    for m in [5, 22, 66, 132, 252]:
        df_calc[f'MA{m}'] = delta.rolling(window=m).mean()
    
    df_calc["Volatilidade"] = delta.rolling(window=5).std()
    
    # Criar o Target (conforme sua regra de 0.5%)
    threshold = 0.005
    df_calc["Target"] = (delta.shift(-1) > (df_calc["Close"] * threshold)).astype(int)
    
    return df_calc.dropna()


if pagina == "📊 Performance do Modelo":
        
 # 🔹 Acurácia em destaque
        st.markdown("### 📊 Performance do Modelo")

        try:
            acc_treino = joblib.load('modelo_xgb_financeiro.joblib').best_score_
            st.metric(
                label="Acurácia Média (Cross-Validation)",
                value=f"{acc_treino * 100:.2f}%",
                delta="XGBoost"
            )
        except:
            st.warning("Acurácia não disponível. Ver script de treino.")

if pagina == "⚙️ Parâmetros do Modelo (XGBoost)":

    xgb_param_descriptions = {
            "n_estimators": "Número de árvores (boosting rounds)",
            "max_depth": "Profundidade máxima de cada árvore",
            "learning_rate": "Taxa de aprendizado (eta)",
            "subsample": "Proporção de amostras usadas por árvore",
            "colsample_bytree": "Proporção de features usadas por árvore",
            "gamma": "Redução mínima de perda para nova divisão",
            "min_child_weight": "Peso mínimo necessário em um nó",
            "reg_alpha": "Regularização L1 (Lasso)",
            "reg_lambda": "Regularização L2 (Ridge)",
            "objective": "Função objetivo do modelo",
            "eval_metric": "Métrica de avaliação",
            }

    clf_params = {k.split('__')[-1]: v for k, v in best_params.items()}

    params_df = pd.DataFrame([
                {
                    "Parâmetro": param,
                    "Valor": value,
                    "Descrição": xgb_param_descriptions.get(
                        param, "Parâmetro interno do XGBoost"
                    )
                }
                for param, value in clf_params.items()
            ])

    st.markdown("### ⚙️ Parâmetros do Modelo (XGBoost)")
    st.dataframe(
                    params_df,
                    use_container_width=True,
                    hide_index=True
                )

if pagina == "🧠 Importância das Features":

# 🔹 Importância das Features
    st.markdown("### 🧠 Importância das Features")

    # Segurança para nomes das colunas
    if hasattr(clf_instance, 'feature_names_in_'):
        nomes_colunas = clf_instance.feature_names_in_
    else:
        nomes_colunas = [
            'Delta', 'Return', 'Delta_lag1', 'Delta_lag2', 'Delta_lag3',
            'Delta_lag4', 'Delta_lag5', 'Delta_lag6', 'Delta_lag7',
            'MA5', 'MA22', 'MA66', 'MA132', 'MA252', 'Volatilidade'
        ]

    if hasattr(clf_instance, 'feature_importances_'):
        feat_imp = pd.DataFrame({
            'Feature': nomes_colunas,
            'Importancia': clf_instance.feature_importances_
        }).sort_values(by='Importancia', ascending=True)

        fig_imp = px.bar(
            feat_imp,
            x='Importancia',
            y='Feature',
            orientation='h',
            template='plotly_dark',
            height=450
        )

        fig_imp.update_layout(
            xaxis_title="Importância",
            yaxis_title="Feature",
            showlegend=False
        )

        st.plotly_chart(fig_imp, use_container_width=True)


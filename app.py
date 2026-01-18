import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import warnings
import yfinance as yf
import joblib
import plotly.express as px
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report





# Configurações iniciais

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Ibovespa Predictor Pro", layout="wide")



# ===================================================== Inicio do Cabeçalho ======================================================================================================
st.markdown("<h1 style='text-align: center;'>📈 Ibovespa Dashboard</h1>", unsafe_allow_html=True)

st.divider()
st.markdown("<p style='text-align: center;'>Insira os dados e visualize previsões do mercado de ações de forma simples e interativa.</p>", unsafe_allow_html=True)

st.divider()
# ===================================================== Fim do Cabeçalho =========================================================================================================

with st.expander("ℹ️ Como funciona este dashboard?", expanded=False):
    st.markdown("""
    ### 📈 O que este app faz?
    
    Este dashboard utiliza **modelos de Machine Learning** treinados com dados históricos
    para **estimar a probabilidade de movimentos de alta** no mercado financeiro.

    🔹 O foco é **análise direcional**, não previsão exata de preços  
    🔹 Os sinais são baseados em **padrões históricos**, não em notícias ou fundamentos  
    🔹 O objetivo é **educacional e exploratório**

    ---
    ### ⚠️ Importante
    - O mercado financeiro envolve riscos
    - Resultados passados **não garantem** resultados futuros
    - Nenhuma decisão deve ser tomada apenas com base neste app
    """)


# Centralizar o texto dentro de todos os inputs de texto

st.markdown("""
    <style>
    input {
        text-align: center;
        text-transform: uppercase;
    }
    </style>
    """, unsafe_allow_html=True)


# ======================================================= Função para buscar lista de tickers =========================================================================

@st.cache_data # Cache para não baixar a lista toda hora


def buscar_lista_tickers():
    # Exemplo: Lista das 10 maiores do IBOV
    return ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA", "ABEV3.SA", "BBAS3.SA", "^BVSP"]

tickers = buscar_lista_tickers()

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    # O usuário pode selecionar da lista ou digitar para filtrar
    ticker = st.selectbox("Escolha um índice ou ativo:", options=tickers)


st.divider()

# ======================================================= Função para buscar lista de tickers =========================================================================

# 1. DEFINIÇÃO DAS FUNÇÕES 

@st.cache_resource

def treinar_modelo_futuro(X_final, y_final):
    final_xgb = XGBClassifier(
        subsample=0.8,
        n_estimators=200,
        max_depth=3,
        learning_rate=0.3,
        random_state=42
    )

    final_model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', final_xgb)
    ])

    final_model.fit(X_final, y_final)
    return final_model

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

    
    # # Extrair os melhores parâmetros e a instância do classificador
    # try:
    #     # Se veio de um RandomizedSearchCV/GridSearchCV
    #     params = model.best_params_
    #     clf_instance = model.best_estimator_.named_steps['clf']
    #     pipeline_obj = model.best_estimator_
    # except:
    #     # Se for o Pipeline direto
    #     params = model.get_params()
    #     clf_instance = model.named_steps['clf']
    #     pipeline_obj = model
        
    # return pipeline_obj, params, clf_instance

model, best_params, clf_instance = load_model()


def preparar_dados(ticker):
    # auto_adjust=True ajuda a manter os nomes consistentes
    df = yf.download(ticker, period="2y", interval="1d", auto_adjust=True)
    
    if df.empty: 
        return None, None
    
    # 1. Se for MultiIndex (comum no yfinance novo), pegamos apenas o primeiro nível
    if isinstance(df.columns, pd.MultiIndex): 
        df.columns = df.columns.get_level_values(0)

    # 2. REMOVER COLUNAS DUPLICADAS (Isso resolve o erro do Narwhals)
    df = df.loc[:, ~df.columns.duplicated()].copy()

    # Criamos o df_calc para as features
    df_calc = pd.DataFrame(index=df.index)
    
    # Verificamos se 'Close' existe (no auto_adjust ele pode vir como 'Close')
    col_fechamento = 'Close' if 'Close' in df.columns else 'Close' 
    df_calc["Close"] = df[col_fechamento]
    
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

# 2. CHAMADA DAS FUNÇÕES E LÓGICA PRINCIPAL ---

input_data, df_novo = preparar_dados(ticker)

if df_novo is not None:
    # Criamos a coluna 'Último' para manter compatibilidade com seu código antigo
    df_novo['Último'] = df_novo['Close']
    
    # Processamento para os cálculos das abas
    df_diferenciada = df_novo[['Último']].copy()
    df_diferenciada.rename(columns={"Último": "Close"}, inplace=True)
    delta = df_diferenciada["Close"].diff()
    
    # Configurações globais
    # 0.005 representa 0,5%
    threshold = 0.005 
    n_future = 5



# 3. ABAS E VISUALIZAÇÃO

    tab1, tab2, tab3 = st.tabs(["📊 Análise Exploratória", "ℹ️ Detalhes do Treinamento e Testes", "🔮 Previsão Futura"])

    with tab1:
        st.info("""

Aqui você visualiza o comportamento recente do ativo selecionado.
O gráfico de candles mostra:
- Preço de abertura
- Máxima e mínima
- Fechamento diário

Essas informações ajudam a contextualizar o sinal gerado pelo modelo.
""")

        # Aqui você coloca o seu código do gráfico candlestick, etc.
                    # 5. Gráfico de Candlestick
        st.subheader(f"Visão de Mercado - {ticker}")
        fig_candle = go.Figure(data=[go.Candlestick(
            x=df_novo.tail(60).index,
            open=df_novo.tail(60)['Open'], 
            high=df_novo.tail(60)['High'],
            low=df_novo.tail(60)['Low'], 
            close=df_novo.tail(60)['Close'],
            name="Candlesticks")])
        
        fig_candle.update_layout(
            template="plotly_dark", 
            height=500, 
            xaxis_rangeslider_visible=False
        )
        st.plotly_chart(fig_candle, use_container_width=True)

        if st.button("Analisar Mercado"):
            with st.spinner('Baixando dados e processando indicadores...'):
                # Usamos a função que já criamos no topo do script
                input_data, df_full = preparar_dados(ticker)
                
                if input_data is not None and not input_data.empty:
                    # model vem do seu load_model() lá no topo do script
                    pred = model.predict(input_data)[0]
                    prob = model.predict_proba(input_data)[0]

                    # Métricas em colunas
                    col_m1, col_m2, col_m3 = st.columns(3)
                    ultimo_preco = df_full['Close'].iloc[-1]
                    # Cálculo seguro da variação
                    if len(df_full) > 1:
                        variacao_dia = (df_full['Close'].iloc[-1] / df_full['Close'].iloc[-2] - 1) * 100
                    else:
                        variacao_dia = 0

                    col_m1.metric("Último Preço", f"R$ {ultimo_preco:,.2f}")
                    col_m2.metric("Variação do Dia", f"{variacao_dia:.2f}%")
                    col_m3.metric("Confiança do Modelo", f"{max(prob)*100:.1f}%")

                    st.divider()
                    
                    # Resultado principal

                    st.caption("""
                    🔎 **Como interpretar o sinal**

                    O modelo não prevê preços.
                    Ele estima a **probabilidade** de o próximo período apresentar
                    uma variação positiva relevante com base em padrões históricos.
""")


                    if pred == 1:
                        st.success(f"### 📈 SINAL DE ALTA DETECTADO\nO modelo estima uma subida superior a 0.5% para o próximo período com {prob[1]*100:.1f}% de probabilidade.")
                    else:
                        st.warning(f"### 📉 SINAL NEUTRO / BAIXA\nO modelo não detectou força para uma subida acima de 0.5%. Probabilidade de Estabilidade/Queda: {prob[0]*100:.1f}%.")
                else:
                    st.error("Erro ao processar dados para o sinal em tempo real.")
                with st.expander("📘 Como interpretar este sinal?"):
                    st.markdown(f"""
                    ### 📈 O que significa este sinal?

                    O modelo identificou padrões históricos que,
                    em situações semelhantes, estiveram associados
                    a uma **probabilidade maior de alta** no próximo período.

                    ### 🎯 Papel do threshold (0,5%)
                    - Movimentos menores que **0,5%** são tratados como ruído
                    - Apenas variações acima desse valor são consideradas relevantes

                    ### 🧠 O que este sinal NÃO significa
                    - Não é recomendação de compra ou venda
                    - Não prevê preços exatos
                    - Não considera notícias, eventos ou fundamentos

                    ### ⚖️ Como usar na prática
                    Utilize este sinal como **apoio à análise**,
                    sempre combinado com gestão de risco e outros indicadores.
                    """)


    with tab2:
        
        st.info("""

O modelo principal utiliza **XGBoost**, um algoritmo baseado em árvores de decisão,
muito usado em aplicações financeiras por sua capacidade de capturar padrões não lineares.

Ele foi treinado com:
- Retornos históricos
- Variações diárias
- Indicadores técnicos derivados do preço
""")

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
            st.caption("""
            🧠 **Importância das features**

            Este gráfico mostra quais variáveis tiveram maior influência nas decisões do modelo.
            Valores maiores indicam maior impacto relativo na previsão.
            """)

            fig_imp.update_layout(
                xaxis_title="Importância",
                yaxis_title="Feature",
                showlegend=False
            )

            st.plotly_chart(fig_imp, use_container_width=True)


else:
    st.error("Aguardando dados ou Ticker inválido. Por favor, insira um código válido do Yahoo Finance.")



with tab3:

    st.info("""
**Previsão Futura (Exploratória)**

A previsão dos próximos dias utiliza um **modelo simplificado** e recursivo.
Ela NÃO representa uma previsão precisa de preços,
servindo apenas para visualizar possíveis tendências.
""")

    st.subheader(f"🔮 Previsão Recursiva para {n_future} Dias")


    # st.info("ℹ️ A previsão futura utiliza um modelo simplificado apenas para fins exploratórios.")


    # 1. Preparação dos dados para o Modelo de Produção (Simplificado para 3 lags)
    # Criando features de lag
    def create_lag_features(series, lags=3):
        data = {f'lag{i}': series.shift(i) for i in range(1, lags + 1)}
        return pd.DataFrame(data)

    X_lagged = create_lag_features(delta, 3)
    # Definindo alvo binário baseado no threshold
    y_binary = pd.Series(np.where(delta > threshold, 1, 0), index=delta.index)

    # Alinhamento e limpeza
    combined = pd.concat([X_lagged, y_binary.rename('target')], axis=1).dropna()
    X_final = combined[['lag1', 'lag2', 'lag3']]
    y_final = combined['target']

    # 2. TREINAMENTO DO MODELO (Isso define o 'final_model')
    final_xgb = XGBClassifier(subsample=0.8, n_estimators=200, max_depth=3, learning_rate=0.3, random_state=42)
    final_model = Pipeline([('scaler', StandardScaler()), ('clf', final_xgb)])
    final_model = treinar_modelo_futuro(X_final, y_final)

    # 3. FUNÇÃO DE FORECAST (Definição)
    def forecast(model, series, steps, thresh):
        preds = []
        temp_series = list(series.dropna().tail(3).values)
        for _ in range(steps):
            vals = temp_series[-3:][::-1] 
            X_next = pd.DataFrame([vals], columns=['lag1', 'lag2', 'lag3'])
            p = model.predict(X_next)[0]
            preds.append(int(p))
            new_val = thresh + 0.01 if p == 1 else -thresh - 0.01
            temp_series.append(new_val)
        return preds

    # 4. EXECUÇÃO DA PREVISÃO (Agora o 'final_model' já existe!)
    f_preds = forecast(final_model, delta, n_future, threshold)

    # 5. CRIAÇÃO DA TABELA DE RESULTADOS
    last_d = delta.index.max()
    future_dates = pd.date_range(start=last_d + pd.Timedelta(days=1), periods=n_future, freq='B')
    
    df_f = pd.DataFrame({
        'Data': future_dates.strftime('%d/%m/%Y'), 
        'Previsão': f_preds
    })
    df_f['Tendência'] = df_f['Previsão'].map({1: "ALTA ▲", 0: "QUEDA ▼"})

    # Função para colorir a tabela
    def colorir_tendencia(val):
        color = '#2ecc71' if 'ALTA' in val else '#e74c3c'
        return f'color: {color}; font-weight: bold'

    st.write("### Tendência para os próximos dias úteis:")
    st.dataframe(
        df_f[['Data', 'Tendência']].style.applymap(colorir_tendencia, subset=['Tendência']),
        use_container_width=True
    )
    with st.expander("📘 Como essa previsão futura é gerada?"):
        st.markdown("""
        - O modelo utiliza apenas **variações passadas do preço**
        - Cada previsão alimenta a próxima (forecast recursivo)
        - Pequenos erros podem se acumular ao longo do tempo

        👉 Por isso, esta funcionalidade deve ser usada apenas
        como **exercício exploratório**.
        """)

# ====================================================== Parte Final ==============================================================================================================
st.divider()

st.markdown(
    """
    <div style="
        background-color: rgba(28, 131, 225, 0.1); 
        border: 1px solid rgba(28, 131, 225, 0.2); 
        padding: 15px; 
        border-radius: 0.5rem; 
        color: #1c83e1; 
        text-align: center; 
        font-size: 0.9rem;
        margin-top: 20px;
        margin-bottom: 20px;
        line-height: 1.6;">
        <strong>Equipe de desenvolvimento</strong><br>
        Eduardo Jorge — Erikson Machado — Mariangela da Silva — Marcos Aurélio — Moacir Carlos
    </div>
    """, 
    unsafe_allow_html=True
)

st.markdown(
    """
    <p style='text-align: center; color: gray; font-size: 0.8rem;'>
        Aviso: Esta é uma ferramenta educacional e não constitui recomendação de investimento.
    </p>
    """, 
    unsafe_allow_html=True
)
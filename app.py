import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Configuração da página
st.set_page_config(
    page_title="Recomendação de Cultivos Agrícolas",
    page_icon="🌱",
    layout="centered"
)

# Título e descrição
st.title("🌾 Recomendação Inteligente de Cultivos")
st.markdown("""
Insira as condições do solo e clima para receber a melhor recomendação de cultivo!
""")

# Carregar modelo (cache para performance)
@st.cache_resource
def load_model():
    pipeline = joblib.load('crop_recommendation_pipeline.joblib')
    encoder = joblib.load('label_encoder.joblib')
    return pipeline, encoder

model, le = load_model()

# Formulário de entrada
with st.form("input_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        n = st.slider("Nitrogênio (N) kg/ha", 0, 150, 90)
        p = st.slider("Fósforo (P) kg/ha", 0, 150, 42)
        k = st.slider("Potássio (K) kg/ha", 0, 150, 43)
        temp = st.slider("Temperatura (°C)", 0.0, 50.0, 20.8)
    
    with col2:
        humidity = st.slider("Umidade Relativa (%)", 0.0, 100.0, 82.0)
        ph = st.slider("pH do Solo", 0.0, 14.0, 6.5)
        rainfall = st.slider("Chuva (mm/ano)", 0.0, 500.0, 203.0)
    
    submitted = st.form_submit_button("Recomendar Cultivo")

# Processamento quando enviado
if submitted:
    try:
        # Criar DataFrame com inputs
        input_data = pd.DataFrame([[n, p, k, temp, humidity, ph, rainfall]], 
                                columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
        
        # Fazer predição
        prediction = model.predict(input_data)
        
        crop = str(prediction[0])
        
        # Mostrar resultado
        st.success(f"### Cultivo Recomendado: {crop.capitalize()}")
        
        # Exibir probabilidades (se disponível)
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(input_data)[0]
            st.subheader("Probabilidades por Cultivo")
            
            # Obter nomes das classes
            if hasattr(le, 'classes_'):
                class_names = le.classes_
            else:
                class_names = [f"Classe {i}" for i in range(len(probs))]
            
            prob_df = pd.DataFrame({
                'Cultivo': class_names,
                'Probabilidade (%)': (probs * 100).round(2)
            }).sort_values('Probabilidade (%)', ascending=False)
            
            st.bar_chart(prob_df.set_index('Cultivo'))
    
    except Exception as e:
        st.error(f"Ocorreu um erro: {str(e)}")

# Seção de informações
st.divider()
st.markdown("""
### Como funciona?
O sistema analisa 7 fatores críticos:
1. Nutrientes do solo (N, P, K)
2. Condições climáticas (temperatura, umidade)
3. pH do solo e precipitação

ℹ️ *Os resultados são recomendações estatísticas. Consulte um agrônomo para decisões finais.*
""")
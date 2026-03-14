import pandas as pd
import streamlit as st
import mlflow
import joblib



@st.cache_resource
def load_model():
    # Carrega o modelo direto do arquivo, sem depender da internet ou do MLflow!
    return joblib.load('modelo_salarios.joblib')

model = load_model()



data_template = pd.read_csv('data/template.csv')
st.markdown("# Data Salary")


col1, col2,col3, col4 = st.columns(4)

with col1:
    idade = st.number_input("Idade",
                        min_value=data_template['idade'].min(),
                        max_value=100)

    genero = st.selectbox("Genero",
                      options= data_template["genero"].unique()
                      )

    pcd = st.selectbox("PCD",
                      options= data_template["pcd"].unique()
                      )
    
    ufs = data_template["ufOndeMora"].sort_values().unique().tolist()
    uf = st.selectbox("UF", options=ufs)
    
with col2:
    cargos = data_template["cargoAtual"].sort_values().unique().tolist()
    cargo_atual = st.selectbox("Cargo Atual", options=cargos)
    
    niveis = data_template["nivel"].sort_values().unique().tolist()
    nivel = st.selectbox("Nível", options=niveis)

with col3:
    temp_dados = data_template["tempoDeExperienciaemDados"].sort_values().unique().tolist()
    temp_exp_dados = st.selectbox("Tempo de Exp. em dados", options=temp_dados)

    temp_ti = data_template["tempoDeExperienciaEmTi"].sort_values().unique().tolist()
    temp_exp_ti = st.selectbox("Tempo de Exp. em TI", options=temp_ti, help='Outras áreas correlacionadas a tecnologia.')

    

data = pd.DataFrame([{
    "idade": idade,
    "genero": genero,   
    "pcd": pcd,       
    "ufOndeMora": uf,
    "cargoAtual": cargo_atual,           
    "nivel": nivel,         
    "tempoDeExperienciaemDados":temp_exp_dados,
    "tempoDeExperienciaEmTi":temp_exp_ti
}])

salario = model.predict(data[model.feature_names_in_])[0]
salario = salario.split("- ")[-1]
st.text(f"Sua faixa salarial é {salario}")
    

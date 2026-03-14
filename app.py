import pandas as pd
import streamlit as st
import mlflow

@st.cache_resource(ttl='1day')
def load_model():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    models =[i for i in  mlflow.search_registered_models() if i.name=="salario-model"]
    last_version = max([int(i.version) for i in models[0].latest_versions])
    model = mlflow.sklearn.load_model(f"models:///salario-model/{last_version}")
    return model

model = load_model()



data_template = pd.read_csv('data/template.csv')
st.markdown("# Data Salary")


col1, col2,col3, col4 = st.columns(4)

with col1:
    idade = st.number_input("idade",
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
    

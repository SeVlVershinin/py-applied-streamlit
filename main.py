import streamlit as st
import pandas as pd

st.set_page_config(layout="wide",
                   page_title="Прогнозирование отклика клиента банка на маркетинговую кампанию",)

st.title('Привет! Я тайтл')
st.subheader('Приветики! А я сабтайтл))')


tab1, tab2, tab3 = st.tabs(["Описание данных", "Анализ", "Модель"])

tab1.header("Описание данных")
tab2.header("Анализ")
tab3.header("Модель")

clients = pd.read_csv('datasets/D_clients.csv')
with tab1:
    st.dataframe(clients)
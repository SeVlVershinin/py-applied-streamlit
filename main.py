import asyncio

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

st.set_page_config(
    layout='wide',
    page_title="Прогнозирование отклика клиента банка на маркетинговую кампанию"
)

st.title('Разведочный анализ данных о клиентах банка')
st.subheader('Приветики! А я сабтайтл))')

bank_image = Image.open('bank_logo.jpg')
st.image(
    bank_image,
    width=800
)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Набор данных",
     "Числовые признаки",
     "Категориальные признаки",
     "Выводы",
     "Предсказания"
     ])

clients = pd.read_csv('datasets/client_full_data.csv')


def render_tab1():
    st.markdown("""
    ##### Набор данных и его описание

    Приложение предназначено для отображения результатов разведочного анализа данных о клиентах банка. Ниже вы можете
    посмотреть эти данные в виде таблицы. Поскольку в набор имеется большое количество столбцов, воспользуйтесь 
    горизонтальной прокруткой
    """)
    st.dataframe(clients)
    st.markdown("""
    ##### Описание признаков
    
    Набор данных включает в себя следующие характеристики клиента:
    """)
    st.markdown("""
        - AGE - возраст клиента;
        - GENDER - пол клиента (1 — мужчина, 0 — женщина);
        - EDUCATION - уровень образования клиента;
        - MARITAL_STATUS - семейное положение клиента;
        - CHILD_TOTAL - количество детей у клиента;
        - DEPENDANT_TOTAL - количество человек на иждивении у клиента;
        - EMPLOYED - трудоустроен ли клиент (1 — работает, 0 — не работает);
        - RETIRED - является ли клиент пенсионером (1 — пенсионер, 0 — не пенсионер);
        - REG_ADDRESS_PROVINCE - регион регистрации клиента;
        - FACT_ADDRESS_PROVINCE - регион фактического пребывания клиента; 
        - POSTAL_ADDRESS_PROVINCE - регион в составе почтового адреса клиента;
        - FLAT_OWNER - является ли клиент владельцем квартиры (1 — есть, 0 — нет);
        - AUTO_TOTAL - количество автомобилей в собственности у клиента;
        - TARGET - (целевая переменная) отклик клиента на маркетинговую кампанию (1 — отклик был, 0 — отклика не было);
        - EMPLOYMENT_INDUSTRY - отрасль в которой трудоустроен клиент;
        - EMPLOYMENT_POSITION - должность клиента;
        - EMPLOYMENT_ROLE - направление деятельности клиента внутри компании;
        - EMPLOYMENT_DURATION - время работы клиента на текущем месте (в месяцах);
        - FAMILY_INCOME - семейный доход клиента (один из нескольких диапазонов);
        - PERSONAL_INCOME - личный доход клиента;
        - RECENT_LOAN_AMOUNT - размер последнего кредита клиента;
        - RECENT_LOAN_LENGTH - срок последнего кредита клиента;
        - RECENT_LOAN_FIRST_PAYMENT - размер первоначального взноса по последнему кредиту клиента;
        - LOAN_COUNT - количество кредитов в кредитной истории клиента;
        - CLOSED_LOAN_COUNT - количество закрытых кредитор в кредитной истории клиента.
        
    """)


def render_tab2():
    num_columns = [
        'AGE',
        'CHILD_TOTAL',
        'DEPENDANT_TOTAL',
        'AUTO_TOTAL',
        'EMPLOYMENT_DURATION',
        'PERSONAL_INCOME',
        'RECENT_LOAN_AMOUNT',
        'RECENT_LOAN_LENGTH',
        'RECENT_LOAN_FIRST_PAYMENT',
        'LOAN_COUNT',
        'CLOSED_LOAN_COUNT'
    ]
    st.markdown("""
            #### Числовые признаки и их анализ 
            
            ##### 1. Состав и характеристики числовых признаков   
            В таблице ниже приведен перечень числовых признаков и их основные характеристики: количество значений, 
            среднее значение и стандартное отклонение, минимум, максимум, медиана и 25/75 перцентили.
            """)
    st.dataframe(clients[num_columns].describe().T, height=430, width=1000)

    st.markdown("""
                ##### 2. Распределения числовых признаков
    
                Посмотрим на графики распределения числовых признаков с большим количеством возможных значений.
                """)

    et1 = st.empty()
    et1.info("Пожалуйста, подождите немного... Формирование графиков почти завершено ... ")

    st.markdown("""
                ##### 3. Распределения числовых признаков с ограниченным дискретным набором значений 
 
                А теперь взглянем на графики распределения числовых признаков с ограниченным набором значений.
                """)

    et2 = st.empty()
    et2.info("Пожалуйста, подождите немного... Формирование графиков почти завершено ... ")

    st.markdown("""
                    ##### 4. Корреляция числовых признаков 

                    А теперь взглянем на графики со значениями коэффициентов корреляции  числовых признаков.
                    """)

    et3 = st.empty()
    et3.info("Пожалуйста, подождите немного... Формирование графиков почти завершено ... ")

    num_columns_cont = [
        'AGE',
        'EMPLOYMENT_DURATION',
        'PERSONAL_INCOME',
        'RECENT_LOAN_AMOUNT',
        'RECENT_LOAN_FIRST_PAYMENT',
    ]

    f = plt.figure(figsize=(20, 10))
    ax = f.subplots(2, 3)
    for i in range(len(num_columns_cont)):
        sns.histplot(data=clients, x=num_columns_cont[i], bins=100, ax=ax[i // 3, i % 3])
    f.delaxes(ax[1, 2])
    et1.pyplot(f, use_container_width=False)

    num_columns_discrete = [
        'CHILD_TOTAL',
        'DEPENDANT_TOTAL',
        'AUTO_TOTAL',
        'LOAN_COUNT',
        'CLOSED_LOAN_COUNT',
        'RECENT_LOAN_LENGTH'
    ]
    f = plt.figure(figsize=(20, 10))
    ax = f.subplots(2, 3)
    for i in range(len(num_columns_discrete)):
        d = clients.groupby(num_columns_discrete[i], as_index=False).agg(COUNT=(num_columns_discrete[i], 'count'))
        sns.barplot(d, x=num_columns_discrete[i], y='COUNT', width=0.5, ax=ax[i // 3, i % 3])
    et2.pyplot(f, use_container_width=False)

    f = plt.figure()
    ax = f.subplots(1, 1)
    sns.heatmap(clients[num_columns_cont + num_columns_discrete].corr(), annot=True, cmap="Blues", ax=ax)
    f.tight_layout()
    # sns.set(font_scale=0.4)
    et3.pyplot(f, use_container_width=False)


with tab1:
    render_tab1()
with tab2:
    render_tab2()

import asyncio

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image

FIG_SIZE = (20, 8)

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
     "Целевая переменная",
     "Предсказания"
     ])

clients = pd.read_csv('datasets/client_full_data.csv')
delayed_render = []


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

    def render_figures_on_tab2():
        num_columns_cont = [
            'AGE',
            'EMPLOYMENT_DURATION',
            'PERSONAL_INCOME',
            'RECENT_LOAN_AMOUNT',
            'RECENT_LOAN_FIRST_PAYMENT',
        ]

        f = plt.figure(figsize=FIG_SIZE)
        ax = f.subplots(2, 3)
        for i in range(len(num_columns_cont)):
            sns.histplot(data=clients, x=num_columns_cont[i], bins=100, ax=ax[i // 3, i % 3])
        f.delaxes(ax[1, 2])
        et1.pyplot(f, use_container_width=False)
        plt.close()

        num_columns_discrete = [
            'CHILD_TOTAL',
            'DEPENDANT_TOTAL',
            'AUTO_TOTAL',
            'LOAN_COUNT',
            'CLOSED_LOAN_COUNT',
            'RECENT_LOAN_LENGTH'
        ]
        f = plt.figure(figsize=FIG_SIZE)
        ax = f.subplots(2, 3)
        for i in range(len(num_columns_discrete)):
            d = clients.groupby(num_columns_discrete[i], as_index=False).agg(COUNT=(num_columns_discrete[i], 'count'))
            sns.barplot(d, x=num_columns_discrete[i], y='COUNT', width=0.5, ax=ax[i // 3, i % 3])
        et2.pyplot(f, use_container_width=False)
        plt.close()

        f = plt.figure(figsize=FIG_SIZE)
        ax = f.subplots(1, 1)
        sns.heatmap(clients[num_columns_cont + num_columns_discrete].corr(), annot=True, cmap="Blues", ax=ax)
        f.tight_layout()
        et3.pyplot(f, use_container_width=False)
        plt.close()

    # delayed_render.append(render_figures_on_tab2)


def render_tab3():
    st.markdown("""
                #### Числовые признаки и их анализ 

                ##### 1. Состав и характеристики числовых признаков   
                В таблице ниже приведен перечень числовых признаков и их основные характеристики: количество значений, 
                среднее значение и стандартное отклонение, минимум, максимум, медиана и 25/75 перцентили.
                """)
    # st.dataframe(clients[num_columns].describe().T, height=430, width=1000)

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

    def render_figures_on_tab3():
        cat_columns = [
            'GENDER',
            'EMPLOYED',
            'RETIRED',
            'FLAT_OWNER',
        ]

        x_labels = [
            ['женщина', 'мужчина'],
            ['безработный', 'работает'],
            ['не на пенсии', 'на пенсии'],
            ["не владеет квартирой", "владеет квартирой"]
        ]

        f = plt.figure(figsize=(20, 4))
        ax = f.subplots(1, 4)
        for i in range(len(cat_columns)):
            d = clients.groupby(cat_columns[i], as_index=False).agg(COUNT=(cat_columns[i], 'count'))
            g = sns.barplot(d, x=cat_columns[i], y='COUNT', width=0.5, ax=ax[i])
            g.set_xticks(range(2))  # <--- set the ticks first
            g.set_xticklabels(x_labels[i])
        f.tight_layout()
        et1.pyplot(f, use_container_width=False)
        plt.close()

        cat_columns = [
            'REG_ADDRESS_PROVINCE',
            'FACT_ADDRESS_PROVINCE',
            'POSTAL_ADDRESS_PROVINCE',
        ]

        f = plt.figure(figsize=(20, 10))
        ax = f.subplots(1, 3)
        for i in range(len(cat_columns)):
            d = clients.groupby(cat_columns[i], as_index=False).agg(COUNT=(cat_columns[i], 'count')).sort_values(
                'COUNT', ascending=False).head(30)
            sns.barplot(d, y=cat_columns[i], x='COUNT', width=0.5, ax=ax[i])
        f.tight_layout()
        et2.pyplot(f, use_container_width=False)
        plt.close()

        cat_columns = [
            'MARITAL_STATUS',
            'EDUCATION',
            'EMPLOYMENT_INDUSTRY',
            'EMPLOYMENT_POSITION',
            'EMPLOYMENT_ROLE',
            'FAMILY_INCOME'
        ]

        f = plt.figure(figsize=(20, 20))
        ax = f.subplots(3, 2)
        for i in range(len(cat_columns)):
            d = clients.groupby(cat_columns[i], as_index=False).agg(COUNT=(cat_columns[i], 'count'))
            sns.barplot(d, y=cat_columns[i], x='COUNT', width=0.5, ax=ax[i // 2, i % 2])
        f.tight_layout()
        et3.pyplot(f, use_container_width=False)
        plt.close()

    delayed_render.append(render_figures_on_tab3)


def render_tab4():
    st.markdown("""
                #### Целевая переменная и ее зависимость от признаков 

                ##### 1. Распределение    
                """)


    et1 = st.empty()
    et1.info("Пожалуйста, подождите немного... Формирование графиков почти завершено ... ")

    f = plt.figure(figsize=(2, 2))
    ax = f.subplots(1, 1)
    d = clients.groupby('TARGET', as_index=False).agg(COUNT=('TARGET', 'count'))
    g = sns.barplot(d, x='TARGET', y='COUNT', width=0.5, ax=ax)
    # g = sns.histplot(clients['TARGET'], discrete=True, ax=ax)
    g.set_xticks(range(2))  # <--- set the ticks first

    g.set_xticklabels(['не откликался', 'откликнулся'])
    # ax.tick_params(axis='both', which='major', labelsize=6)
    # f.tight_layout()
    et1.pyplot(f, use_container_width=False)
    plt.close()

    st.markdown("""
                    ##### 2. Корреляция 

                    А теперь взглянем на графики распределения числовых признаков с ограниченным набором значений.
                    """)

    et2 = st.empty()
    et2.info("Пожалуйста, подождите немного... Формирование графиков почти завершено ... ")

    st.markdown("""
                        ##### 3. Корреляция числовых признаков 

                        А теперь взглянем на графики со значениями коэффициентов корреляции  числовых признаков.
                        """)

    et3 = st.empty()
    et3.info("Пожалуйста, подождите немного... Формирование графиков почти завершено ... ")

    def render_figures_on_tab4():
        cat_columns = [
            'GENDER',
            'EMPLOYED',
            'RETIRED',
            'FLAT_OWNER',
        ]

        x_labels = [
            ['женщина', 'мужчина'],
            ['безработный', 'работает'],
            ['не на пенсии', 'на пенсии'],
            ["не владеет квартирой", "владеет квартирой"]
        ]

        f = plt.figure(figsize=(20, 4))
        ax = f.subplots(1, 4)
        for i in range(len(cat_columns)):
            d = clients.groupby(cat_columns[i], as_index=False).agg(COUNT=(cat_columns[i], 'count'))
            g = sns.barplot(d, x=cat_columns[i], y='COUNT', width=0.5, ax=ax[i])
            g.set_xticks(range(2))  # <--- set the ticks first
            g.set_xticklabels(x_labels[i])
        f.tight_layout()
        et1.pyplot(f, use_container_width=False)
        plt.close()

        cat_columns = [
            'REG_ADDRESS_PROVINCE',
            'FACT_ADDRESS_PROVINCE',
            'POSTAL_ADDRESS_PROVINCE',
        ]

        f = plt.figure(figsize=(20, 10))
        ax = f.subplots(1, 3)
        for i in range(len(cat_columns)):
            d = clients.groupby(cat_columns[i], as_index=False).agg(COUNT=(cat_columns[i], 'count')).sort_values(
                'COUNT', ascending=False).head(30)
            sns.barplot(d, y=cat_columns[i], x='COUNT', width=0.5, ax=ax[i])
        f.tight_layout()
        et2.pyplot(f, use_container_width=False)
        plt.close()

        cat_columns = [
            'MARITAL_STATUS',
            'EDUCATION',
            'EMPLOYMENT_INDUSTRY',
            'EMPLOYMENT_POSITION',
            'EMPLOYMENT_ROLE',
            'FAMILY_INCOME'
        ]

        f = plt.figure(figsize=(20, 20))
        ax = f.subplots(3, 2)
        for i in range(len(cat_columns)):
            d = clients.groupby(cat_columns[i], as_index=False).agg(COUNT=(cat_columns[i], 'count'))
            sns.barplot(d, y=cat_columns[i], x='COUNT', width=0.5, ax=ax[i // 2, i % 2])
        f.tight_layout()
        et3.pyplot(f, use_container_width=False)
        plt.close()

    # delayed_render.append(render_figures_on_tab4)

with tab1:
    render_tab1()
with tab2:
    render_tab2()
with tab3:
    render_tab3()
with tab4:
    render_tab4()

for r in delayed_render:
    r()

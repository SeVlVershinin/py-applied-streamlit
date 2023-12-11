import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from streamlit.delta_generator import DeltaGenerator

from dataset_metadata import CAT_COLUMNS_OTHER, CAT_COLUMNS_ADDRESS, LABELS_FOR_CAT_COLUMNS_BINARY, CAT_COLUMNS_BINARY, \
    NUM_COLUMNS_DISCRETE, NUM_COLUMNS_CONTINUOUS, NUM_COLUMNS, NUM_COLUMNS_CONTINUOUS_BIN_COUNT

st.set_page_config(
    layout='wide',
    page_title="Прогнозирование отклика клиента банка на маркетинговую кампанию"
)

st.title('Клиенты банка и их отклики на предложения об услугах ')
st.subheader('Разведочный анализ данных')

bank_image = Image.open('bank_logo.jpg')
st.image(
    bank_image,
    width=400
)
st.markdown("""
    В приложении вы найдете результаты разведочного анализа данных о клиентах банка и их откликах на предложения 
    банка о новых услугах. 
    Информация размещена на нескольких вкладках.
    - __Набор данных__  - содержит данные в табличном виде и описание столбцов.
    - __Числовые признаки__ - включает описание и анализ числовых признаков.
    - __Категориальные признаки__ - посвящена описанию и анализ категориальных признаков.
    - __Целевая переменная__ - представляет результаты анализа влияния различных признаков на целевую переменную.   
    - __Выводы__ - подводит итоги анализа и содержит основные выводы. 
    """)

clients = pd.read_csv('datasets/client_full_data.csv')
delayed_render = []


def render_tab1():
    st.markdown("""
    ##### Набор данных и его описание

    Ниже вы можете просмотреть данные о клиентах банка и об их реакции на предложение о новой услуге (целевая переменная)
    в табличном виде. Кликая по заголовкам столбцов, можно отсортировать данные по убыванию или возрастанию. Также можно
    выполнить поиск значений по подстроке или скачать весь набор в виде csv-файла. 
     
    В наборе большое количество столбцов - воспользуйтесь горизонтальной прокруткой
    """)
    st.dataframe(clients)
    st.markdown("""
    ##### Описание признаков
    
    Набор данных включает в себя следующие характеристики клиента:
    """)
    st.markdown(
        """
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
        """
    )


def render_tab2():
    st.markdown(
        """
        #### Числовые признаки и их анализ 
        
        ##### 1. Состав и характеристики числовых признаков   
        
        Ниже приведен перечень числовых признаков датасета и их основные характеристики: количество значений, 
        среднее значение и стандартное отклонение, минимум, максимум, медиана и 25/75 перцентили. Уже по этой таблице 
        мы видим, что всего записей 15197, а в столбцах отсутствуют пропуски (набор был предварительно подготовлен). 
        Также можно заметить, что у большинства признаков медиана (50 перцентиль) сильно отличается от среднего значения,
        что может говорить о том, что их распределение отличает от нормального.
        """
    )
    st.dataframe(clients[NUM_COLUMNS].describe().T, height=430, width=1000)

    st.markdown(
        """
        ##### 2. Распределения числовых признаков
        
        Теперь посмотрим на графики распределения числовых признаков c большим количеством значений (можно сказать, 
        непрерывных). По ним можно сделать следующие выводы:  
        - возраст клиентов в выборке находится в диапазоне от ~20 до ~70 лет, а на диапазоне от ~25 лет количество 
        клиентов в выборке убывает с ростом возраста; 
        - большая часть клиентов выборки работают на своем рабочем месте не более 50 месяцев (т.е. не более ~ 4 лет); 
        - большая часть клиентов выборки имеет личный доход в диапазоне от 5 до 10 тысяч; с ростом значения 
        данного показателя количество клиентов в выборке уменьшается; 
        - количество клиентов в выборке тем больше, чем меньше размер их последнего кредита и величина первоначального 
        взноса по нему.
        """
    )

    et1 = st.empty()
    et1.info("Пожалуйста, подождите немного... Формирование графиков почти завершено ... ")

    st.markdown(
        """
        ##### 3. Распределения числовых признаков с ограниченным дискретным набором значений 
        
        Также построим графики распределения дискретных числовых признаков. Глядя на них, можем отметить, что:  
        - для меньших значений количества детей, человек на иждивении, автомобилей в собственности, а также кредитов (в 
        том числе, закрытых) в выборке присутствует большее количество клиентов; 
        - среди набора типовых сроков кредита наиболее популярны (большое количество клиентов присутствует в выборке)
        сроки в 3, 6, 10 и 12 месяцев.
        """
    )

    et2 = st.empty()
    et2.info("Пожалуйста, подождите немного... Формирование графиков почти завершено ... ")

    st.markdown(
        """
        ##### 4. Корреляция числовых признаков 

        Наконец, посмотрим на значения коэффициентов корреляции между числовыми признаками и отметим следующие 
        моменты: 
        - некоторая положительная корреляция наблюдается между возрастом клиента и количеством детей, а также между 
        возрастом и длительностью работы, что в целом объяснимо, т.к. с возрастом число детей может только расти, как 
        может расти и стаж работы (в том числе, на одном месте). В то же время, наблюдается отрицательная корреляция 
        между возрастом и количеством человек на иждивении, что тоже понятно, т.к. с возрастом дети становятся старше 
        и начинают самостоятельную жизнь;
        - заметная положительная корреляция наблюдается между размером дохода и количеством автомобилей, а также 
        размерами кредита и первого платежа по нему. В свою очередь, размер кредита имеет положительную корреляцию 
        с его длительностью;  
        - ожидаемая положительная корреляция есть между количеством детей и количеством человек на иждивении; 
        - сильная положительная корреляция также имеет место между общим и закрытым количествами кредитов, что говорит о 
        том, что чем больше кредитов берет человек, тем больше их закрывает. 
        """
    )

    et3 = st.empty()
    et3.info("Пожалуйста, подождите немного... Формирование графиков почти завершено ... ")

    def draw_continuous_num_columns_distribution(container: DeltaGenerator):


        f = plt.figure(figsize=(20, 8))
        ax = f.subplots(2, 3)
        for i in range(len(NUM_COLUMNS_CONTINUOUS)):
            sns.histplot(data=clients,
                         x=NUM_COLUMNS_CONTINUOUS[i],
                         bins=NUM_COLUMNS_CONTINUOUS_BIN_COUNT[i],
                         ax=ax[i // 3, i % 3])
        f.delaxes(ax[1, 2])
        container.pyplot(f, use_container_width=False)
        plt.close()

    def draw_discrete_num_columns_distribution(container: DeltaGenerator):
        f = plt.figure(figsize=(20, 8))
        ax = f.subplots(2, 3)
        for i in range(len(NUM_COLUMNS_DISCRETE)):
            d = clients.groupby(NUM_COLUMNS_DISCRETE[i], as_index=False).agg(COUNT=(NUM_COLUMNS_DISCRETE[i], 'count'))
            sns.barplot(d, x=NUM_COLUMNS_DISCRETE[i], y='COUNT', width=0.5, ax=ax[i // 3, i % 3])
        container.pyplot(f, use_container_width=False)
        plt.close()

    def draw_correlation_heatmap(container: DeltaGenerator):
        f = plt.figure(figsize=(20, 8))
        ax = f.subplots(1, 1)
        sns.heatmap(clients[NUM_COLUMNS_CONTINUOUS + NUM_COLUMNS_DISCRETE].corr(), annot=True, cmap="Blues", ax=ax)
        f.tight_layout()
        container.pyplot(f, use_container_width=False)
        plt.close()

    def render_figures():
        draw_continuous_num_columns_distribution(et1)
        draw_discrete_num_columns_distribution(et2)
        draw_correlation_heatmap(et3)

    delayed_render.append(render_figures)


def render_tab3():
    st.markdown(
        """
        #### Категориальные признаки и их анализ 

        ##### 1. Распределения бинарных признаков 

        Посмотрим на графики распределения числовых признаков с большим количеством возможных значений.
        """
    )

    et1 = st.empty()
    et1.info("Пожалуйста, подождите немного... Формирование графиков почти завершено ... ")

    st.markdown(
        """
        ##### 2. Распределения регионов регистрации и проживания клиентов 

        А теперь взглянем на графики распределения числовых признаков с ограниченным набором значений.
        """
    )

    et2 = st.empty()
    et2.info("Пожалуйста, подождите немного... Формирование графиков почти завершено ... ")

    st.markdown(
        """
        ##### 3. Распределение оставшихся категориальных признаков 

        А теперь взглянем на графики со значениями коэффициентов корреляции  числовых признаков.
        """
    )

    et3 = st.empty()
    et3.info("Пожалуйста, подождите немного... Формирование графиков почти завершено ... ")

    def draw_binary_cat_columns_distribution(container: DeltaGenerator):
        f = plt.figure(figsize=(20, 4))
        ax = f.subplots(1, 4)
        for i in range(len(CAT_COLUMNS_BINARY)):
            d = clients.groupby(CAT_COLUMNS_BINARY[i], as_index=False).agg(COUNT=(CAT_COLUMNS_BINARY[i], 'count'))
            g = sns.barplot(d, x=CAT_COLUMNS_BINARY[i], y='COUNT', width=0.5, ax=ax[i])
            g.set_xticks(range(2))  # <--- set the ticks first
            g.set_xticklabels(LABELS_FOR_CAT_COLUMNS_BINARY[i])
        f.tight_layout()
        container.pyplot(f, use_container_width=False)
        plt.close()

    def draw_address_cat_columns_distribution(container: DeltaGenerator):
        f = plt.figure(figsize=(20, 10))
        ax = f.subplots(1, 3)
        for i in range(len(CAT_COLUMNS_ADDRESS)):
            d = clients.groupby(CAT_COLUMNS_ADDRESS[i], as_index=False).agg(
                COUNT=(CAT_COLUMNS_ADDRESS[i], 'count')).sort_values(
                'COUNT', ascending=False).head(30)
            sns.barplot(d, y=CAT_COLUMNS_ADDRESS[i], x='COUNT', width=0.5, ax=ax[i])
        f.tight_layout()
        container.pyplot(f, use_container_width=False)
        plt.close()

    def draw_other_cat_columns_distribution(container: DeltaGenerator):
        f = plt.figure(figsize=(20, 20))
        ax = f.subplots(3, 2)
        for i in range(len(CAT_COLUMNS_OTHER)):
            d = clients.groupby(CAT_COLUMNS_OTHER[i], as_index=False).agg(COUNT=(CAT_COLUMNS_OTHER[i], 'count'))
            sns.barplot(d, y=CAT_COLUMNS_OTHER[i], x='COUNT', width=0.5, ax=ax[i // 2, i % 2])
        f.tight_layout()
        container.pyplot(f, use_container_width=False)
        plt.close()

    def render_figures():
        draw_binary_cat_columns_distribution(et1)
        draw_address_cat_columns_distribution(et2)
        draw_other_cat_columns_distribution(et3)

    delayed_render.append(render_figures)


def render_tab4():
    st.markdown(
        """
        #### Целевая переменная и ее зависимость от признаков 

        ##### 1. Распределение целевой переменной и распределения непрерывных числовых признаков в зависимости
        от целевой переменной   
        """
    )

    et1 = st.empty()
    et1.info("Пожалуйста, подождите немного... Формирование графиков почти завершено ... ")

    st.markdown(
        """
        ##### 2. Распределение дискретных числовых признаков в зависимости от целевой переменной  

        А теперь взглянем на графики распределения числовых признаков с ограниченным набором значений.
        """
    )

    et2 = st.empty()
    et2.info("Пожалуйста, подождите немного... Формирование графиков почти завершено ... ")

    st.markdown(
        """
        ##### 3. Распределение категориальных признаков в зависимости от целевой переменной 

        А теперь взглянем на графики со значениями коэффициентов корреляции  числовых признаков.
        """
    )

    et3 = st.empty()
    et3.info("Пожалуйста, подождите немного... Формирование графиков почти завершено ... ")
    et4 = st.empty()
    et4.info("Пожалуйста, подождите немного... Формирование графиков почти завершено ... ")
    et5 = st.empty()
    et5.info("Пожалуйста, подождите немного... Формирование графиков почти завершено ... ")

    def draw_continuous_num_columns_distribution_by_target(container: DeltaGenerator):
        f = plt.figure(figsize=(20, 6))
        ax = f.subplots(2, 3)
        for i in range(len(NUM_COLUMNS_CONTINUOUS)):
            sns.kdeplot(data=clients, x=NUM_COLUMNS_CONTINUOUS[i], multiple='fill', hue='TARGET',
                        ax=ax[(i + 1) // 3, (i + 1) % 3])

        d = clients.groupby('TARGET', as_index=False).agg(COUNT=('TARGET', 'count'))
        g = sns.barplot(d, x='TARGET', y='COUNT', width=0.5, hue='TARGET', ax=ax[0, 0])
        g.set_xticks(range(2))
        g.set_xticklabels(['не откликался', 'откликнулся'])

        f.tight_layout()
        container.pyplot(f, use_container_width=False)
        plt.close()

    def draw_discrete_num_columns_distribution_by_target(container: DeltaGenerator):
        f = plt.figure(figsize=(20, 6))
        ax = f.subplots(2, 3)
        for i in range(len(NUM_COLUMNS_DISCRETE)):
            d = clients.groupby(NUM_COLUMNS_DISCRETE[i], as_index=False).agg(COUNT=(NUM_COLUMNS_DISCRETE[i], 'count'),
                                                                             ACCEPTED=('TARGET', 'sum'))
            d['LEAD_SHARE'] = d['ACCEPTED'] / d['COUNT']
            sns.barplot(d, x=NUM_COLUMNS_DISCRETE[i], y='LEAD_SHARE', width=0.5, ax=ax[i // 3, i % 3],
                        color='darkorange')

        f.tight_layout()
        container.pyplot(f, use_container_width=False)
        plt.close()

    def draw_binary_cat_columns_distribution_by_target(container: DeltaGenerator):
        f = plt.figure(figsize=(20, 4))
        ax = f.subplots(1, 4)
        for i in range(len(CAT_COLUMNS_BINARY)):
            d = clients.groupby(CAT_COLUMNS_BINARY[i], as_index=False).agg(COUNT=(CAT_COLUMNS_BINARY[i], 'count'),
                                                                           ACCEPTED=('TARGET', 'sum'))
            d['LEAD_SHARE'] = d['ACCEPTED'] / d['COUNT']
            g = sns.barplot(d, x=CAT_COLUMNS_BINARY[i], y='LEAD_SHARE', width=0.5, ax=ax[i],
                            color='darkorange')
            g.set_xticks(range(2))  # <--- set the ticks first
            g.set_xticklabels(LABELS_FOR_CAT_COLUMNS_BINARY[i])
        f.tight_layout()
        container.pyplot(f, use_container_width=False)
        plt.close()

    def draw_address_columns_distribution_by_target(container: DeltaGenerator):
        f = plt.figure(figsize=(20, 10))
        ax = f.subplots(1, 3)
        for i in range(len(CAT_COLUMNS_ADDRESS)):
            def gr(x):
                return pd.Series({
                    'COUNT': x['TARGET'].count(),
                    'ACCEPTED': x['TARGET'].sum(),
                    'LEAD_SHARE': x['TARGET'].sum() / x['TARGET'].count()
                })

            d = (clients.groupby(CAT_COLUMNS_ADDRESS[i], as_index=False)
                 .apply(gr)
                 .sort_values('LEAD_SHARE', ascending=False)
                 .head(40))

            # d = (clients.groupby(CAT_COLUMNS_ADDRESS[i], as_index=False)
            #      .agg(COUNT=(CAT_COLUMNS_ADDRESS[i], 'count'), ACCEPTED=('TARGET', 'sum'))
            #      .sort_values('ACCEPTED', ascending=False)
            #      .head(40))
            # d['LEAD_SHARE'] = d['ACCEPTED'] / d['COUNT']
            sns.barplot(d, y=CAT_COLUMNS_ADDRESS[i], x='LEAD_SHARE', width=0.5, ax=ax[i],
                        color='darkorange')
        f.tight_layout()
        container.pyplot(f, use_container_width=False)
        plt.close()

    def draw_other_cat_columns_distribution_by_target(container: DeltaGenerator):
        f = plt.figure(figsize=(20, 20))
        ax = f.subplots(3, 2)
        for i in range(len(CAT_COLUMNS_OTHER)):
            d = clients.groupby(CAT_COLUMNS_OTHER[i], as_index=False).agg(COUNT=(CAT_COLUMNS_OTHER[i], 'count'),
                                                                          ACCEPTED=('TARGET', 'sum'))
            d['LEAD_SHARE'] = d['ACCEPTED'] / d['COUNT']
            sns.barplot(d, y=CAT_COLUMNS_OTHER[i], x='LEAD_SHARE', width=0.5, ax=ax[i // 2, i % 2], color='darkorange')
        f.tight_layout()
        container.pyplot(f, use_container_width=False)
        plt.close()

    def render_figures():
        draw_continuous_num_columns_distribution_by_target(et1)
        draw_discrete_num_columns_distribution_by_target(et2)

        draw_binary_cat_columns_distribution_by_target(et3)
        draw_address_columns_distribution_by_target(et4)
        draw_other_cat_columns_distribution_by_target(et5)

    delayed_render.append(render_figures)


tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["НАБОР ДАННЫХ",
     "ЧИСЛОВЫЕ ПРИЗНАКИ",
     "КАТЕГОРИАЛЬНЫЕ ПРИЗНАКИ",
     "ЦЕЛЕВАЯ ПЕРЕМЕННАЯ",
     "ВЫВОДЫ"
     ])

with tab1:
    render_tab1()
with tab2:
    render_tab2()
with tab3:
    render_tab3()
with tab4:
    render_tab4()

for render_function in delayed_render:
    render_function()

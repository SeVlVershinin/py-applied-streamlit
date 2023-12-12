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

clients = pd.read_csv('client_full_data.csv')
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
        - PERSONAL_INCOME - личный доход клиента (в рублях);
        - RECENT_LOAN_AMOUNT - размер последнего кредита клиента (в рублях);
        - RECENT_LOAN_LENGTH - срок последнего кредита клиента;
        - RECENT_LOAN_FIRST_PAYMENT - размер первоначального взноса по последнему кредиту клиента (в рублях);
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
        ##### 2. Распределения непрерывных числовых признаков
        
        Теперь посмотрим на графики распределения числовых признаков c большим количеством значений (можно сказать, 
        непрерывных). По ним можно сделать следующие выводы:  
        - возраст клиентов в выборке находится в диапазоне от ~20 до ~70 лет, а на диапазоне от ~25 лет количество 
        клиентов в выборке убывает с ростом возраста; 
        - большая часть клиентов выборки работают на своем рабочем месте не более 50 месяцев (т.е. не более ~ 4 лет); 
        - большая часть клиентов выборки имеет личный доход в диапазоне от 5 до 15-20 тысяч рублей (с учетом текущих 
        уровней цен это довольно мало; возможно, большая часть клиентов в выборке не имеет возможности документально 
        подтвердить свой фактический доход); с ростом значения данного показателя количество клиентов в выборке 
        уменьшается; 
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

        Построим гистограммы, показывающие количество клиентов выборки, имеющих то или иное значение соответствующего
        признака. Из них видно, что:
        - мужчин в выборке больше, чем женщин;
        - количество работающих клиентов и клиентов не на пенсии существенно больше, чем безработных и пенсионеров; 
        - среди клиентов выборки больше тех, кто не владеет квартирой. 
        """
    )

    et1 = st.empty()
    et1.info("Пожалуйста, подождите немного... Формирование графиков почти завершено ... ")

    st.markdown(
        """
        ##### 2. Распределения регионов регистрации и проживания клиентов 

        Посмотрим также на то, как клиенты распределены по регионам регистрации, фактического проживания и регионам,  
        указанным в почтовом адресе. Т.к. регионов много, отсортируем их по количеству клиентов в выборке и выведем
        данные для первых 30. Видим, что клиенты распределены по регионам неравномерно. Например, клиентов из 
        Кемеровской области в выборке примерно вдвое больше, чем из Татарстана.
        """
    )

    et2 = st.empty()
    et2.info("Пожалуйста, подождите немного... Формирование графиков почти завершено ... ")

    st.markdown(
        """
        ##### 3. Распределение оставшихся категориальных признаков 

        Также взглянем на распределение оставшихся категориальных признаков. Из графиков заметим, что: 
        - в выборке больше клиентов, состоящих в браке; 
        - в выборке больше клиентов, имеющих среднее специальное образование (при этом, подавляющее большинство 
        клиентов имеет как минимум оконченное среднее);
        - наиболее часто встречающаяся у клиентов сфера занятости - торговля, а позиции - рабочий и специалист; в 
        качестве роли у большинства клиентов указано участие в основной деятельности, что, вообще говоря, не дает 
        особенного понимания, в чем именно заключается роль человека;  
        - у большинства клиентов семейный доход отнесен к категориям "от 10 до 20 тыс.руб." и "от 20 до 50 тыс.руб".
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
        #### Анализ целевой переменной 

        ##### 1. Распределение целевой переменной и зависимость от непрерывных числовых признаков
        Ниже представлена гистограмма распределения целевой переменной, показывающая соотношения клиентов выборки, 
        отреагировавших на предложения банка (будем также называть их __лидами__, как это принято в маркетинге) и не 
        проявивших к нему интерес. А также диаграммы, демонстрирующие зависимость целевой переменной от непрерывных 
        числовых признаков. Диаграммы отображают долю лидов для тех различных значений соответствующих признаков. 
        
        Можно отметить, что:
        - доля клиентов, откликнувшихся на предложение банка - всего 11,9%;
        - чем старше клиенты, тем меньше среди них лидов;
        - чем дольше клиент работает, тем менее вероятен его отклик;
        - в среднем чем больше доход клиента, тем более частым является отклик (форма графика может объясняться тем, 
        что значения дохода заполняют не весь диапазон значений);
        - размер последнего кредита и первоначального взноса по нему не оказывает сильного влияния на частоту отклика;
        - существенное увеличение доли лидов на краях некоторых графиков, возможно, связано с небольшим количеством 
        клиентов в соответствующих диапазонах значений.   
            
        """
    )

    et1 = st.empty()
    et1.info("Пожалуйста, подождите немного... Формирование графиков почти завершено ... ")

    st.markdown(
        """
        ##### 2. Зависимость целевой переменной от дискретных числовых признаков  

        Посмотрим также на то, как на целевую переменную влияют значения дискретных числовых признаков. Для этого 
        построим для каждого из них гистограмму, на которой каждому значению будет соответствовать доля лидов среди
        клиентов с таким значением признака.
        
        Из графиков можно заметить, что: 
        - с ростом количества детей и количества человек на иждивении доля лидов несколько увеличивается; также можно 
        было бы отметить, что клиенты с 10 детьми откликались на предложения банка в несколько раз чаще остальных, но 
        таких клиентов в выборке совсем мало, с чем и связано такое значение; 
        - количество автомобилей клиента не сильно влияет на частоту отклика (учитывая, что клиентов с 2 автомобилям 
        в выборке мало, делать вывод о том, что владельцы 2 автомобилей почти не откликаются, кажется преждевременным);
        - с ростом количества кредитов (в том числе, закрытых) частота откликов падает;
        - чем больше срок последнего кредита, тем больше доля лидов в выборке. 
         
        """
    )

    et2 = st.empty()
    et2.info("Пожалуйста, подождите немного... Формирование графиков почти завершено ... ")

    st.markdown(
        """
        ##### 3. Зависимость целевой переменной от бинарных категориальных признаков 

        Теперь взглянем, как на целевую переменную влияют значения бинарных категориальных признаков. Для этого постоим
        аналогичные гистограммы, показывающие долю лидов в каждой категории. 
        
        Из графиков видно, что:
        - пол клиента влияет на целевую переменную довольно слабо; 
        - доли лидов в категории работающих и не пенсионеров существенно больше, чем среди безработных и пенсионеров 
        соответственно; 
        - владение квартирой практически не влияет на склонность клиентов к отклику. 
        """
    )

    et3 = st.empty()
    et3.info("Пожалуйста, подождите немного... Формирование графиков почти завершено ... ")

    st.markdown(
        """
        ##### 4. Зависимость целевой переменной от региона регистрации и фактического проживания 

        Выясним, как распределена склонность к отклику на предложения банка по регионам. Для этого построим аналогичные 
        гистограммы с долями лидов в каждом регионе (так как регионов много, отобразим те 40, для которых эта доля 
        максимальна). По графикам видно, что: 
        - доля лидов в разных регионах отличается существенно. Так частота отклика почти вдвое больше среди клиентов, 
        зарегистрированных в Москве, чем среди клиентов, зарегистрированных в Московской области;
        - значение имеет не просто регион, но и то, зарегистрирован ли в нем клиент или реально проживает. Например, 
        мы видим, что доля лидов, среди клиентов, фактически проживающих в Дагестане - почти 95%, в то время, как в 
        качестве региона регистрации он не попал даже в топ 40.    
        """
    )

    et4 = st.empty()
    et4.info("Пожалуйста, подождите немного... Формирование графиков почти завершено ... ")

    st.markdown(
        """
        ##### 5. Зависимость целевой переменной от остальных категориальных признаков 

        Наконец, оценим влияние на целевую переменную оставшихся категориальных признаков, построив аналогичные 
        гистограммы распределения доли лидов по категориям. 
        
        Здесь мы видим, что:
        - семейное положение влияет на количество откликов, но не сильно; 
        - клиенты с двумя и более высшими образованиями, а также клиенты с неоконченным высшим образованием 
        откликались на предложения ощутимо чаще; 
        - лидером по доле лидов среди отраслей занятости клиентов является недвижимость; и в целом данная категория 
        влияет на склонность к откликам;
        - среди ролей лидером по частоте откликов является реклама и маркетинг, что, возможно, связано со спецификой 
        профессиональной деятельности;  
        - семейный доход имеет некоторое влияние на количество лидов, но оно не сильно выражено; исключением является 
        категория доходов свыше 50 тыс.руб., но это может быть связано с небольшим количество таких клиентов в выборке.
        """
    )

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
        for i in range(2):
            g.text(i, d['COUNT'][i],
                   f'{(d["COUNT"][i] / d["COUNT"].sum()):.2%}',
                   fontsize=12, verticalalignment='bottom', horizontalalignment='center')
        g.set_ylim(0, 15000)
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

#!/usr/bin/env python
# coding: utf-8

# **Задание 1. Импортируйте библиотеку pandas. Считайте данные из csv-файла в датафрейм и сохраните в переменную
# `data`. Путь к файлу:**

import pandas as pd

try:
    data = pd.read_csv('/datasets/data.csv')
except:
    data = pd.read_csv('https://code.s3.yandex.net/datasets/data.csv')

# **Задание 2. Выведите первые 20 строчек датафрейма `data` на экран.**

print("ex-2\n", data.head(20))

# **Задание 3. Выведите основную информацию о датафрейме с помощью метода `info()`.**

data.info()

# **Задание 4. Выведите количество пропущенных значений для каждого столбца. Используйте комбинацию двух методов.**

print("ex-4\n", data.isna().sum())

# **Задание 5. В двух столбцах есть пропущенные значения. Один из них — `days_employed`. Пропуски в этом столбце вы
# обработаете на следующем этапе. Другой столбец с пропущенными значениями — `total_income` — хранит данные о
# доходах. На сумму дохода сильнее всего влияет тип занятости, поэтому заполнить пропуски в этом столбце нужно
# медианным значением по каждому типу из столбца `income_type`. Например, у человека с типом занятости `сотрудник`
# пропуск в столбце `total_income` должен быть заполнен медианным доходом среди всех записей с тем же типом.**

for t in data['income_type'].unique():
    data.loc[(data['income_type'] == t) & (data['total_income'].isna()),
             'total_income'] = data.loc[(data['income_type'] == t), 'total_income'].median()

# **Задание 6. В данных могут встречаться артефакты (аномалии) — значения, которые не отражают действительность и
# появились по какой-то ошибке. Таким артефактом будет отрицательное количество дней трудового стажа в столбце
# `days_employed`. Для реальных данных это нормально. Обработайте значения в этом столбце: замените все отрицательные
# значения положительными с помощью метода `abs()`.**

data['days_employed'] = data['days_employed'].abs()

# **Задание 7. Для каждого типа занятости выведите медианное значение трудового стажа `days_employed` в днях.**

print("ex-7\n", data.groupby('income_type')['days_employed'].agg('median'))

# У двух типов (безработные и пенсионеры) получатся аномально большие значения. Исправить такие значения сложно,
# поэтому оставьте их как есть. Тем более этот столбец не понадобится вам для исследования.

# **Задание 8. Выведите перечень уникальных значений столбца `children`.**

print("ex-8\n", data['children'].unique())

# **Задание 9. В столбце `children` есть два аномальных значения. Удалите строки, в которых встречаются такие
# аномальные значения из датафрейма `data`.**

data = data[(data['children'] != -1) & (data['children'] != 20)]

# **Задание 10. Ещё раз выведите перечень уникальных значений столбца `children`, чтобы убедиться, что артефакты
# удалены.**

print("ex-10\n", data['children'].unique())

# **Задание 11. Заполните пропуски в столбце `days_employed` медианными значениями по каждого типа занятости
# `income_type`.**

for t in data['income_type'].unique():
    data.loc[(data['income_type'] == t) & (data['days_employed'].isna()),
             'days_employed'] = data.loc[(data['income_type'] == t), 'days_employed'].median()

# **Задание 12. Убедитесь, что все пропуски заполнены. Проверьте себя и ещё раз выведите количество пропущенных
# значений для каждого столбца с помощью двух методов.**

print("ex-12\n", data.isna().sum())

# **Задание 13. Замените вещественный тип данных в столбце `total_income` на целочисленный с помощью метода `astype(
# )`.**

data['total_income'] = data['total_income'].astype(int)

# **Задание 14. Обработайте неявные дубликаты в столбце `education`. В этом столбце есть одни и те же значения,
# но записанные по-разному: с использованием заглавных и строчных букв. Приведите их к нижнему регистру. Проверьте
# остальные столбцы.**

data['education'] = data['education'].str.lower()

# **Задание 15. Выведите на экран количество строк-дубликатов в данных. Если такие строки присутствуют, удалите их.**

print("ex-15\n", data.duplicated().sum())

data = data.drop_duplicates()


# **Задание 16. На основании диапазонов, указанных ниже, создайте в датафрейме `data` столбец `total_income_category`
# с категориями:**
# 
# - 0–30000 — `'E'`;
# - 30001–50000 — `'D'`;
# - 50001–200000 — `'C'`;
# - 200001–1000000 — `'B'`;
# - 1000001 и выше — `'A'`.
# 
# 
# **Например, кредитополучателю с доходом 25000 нужно назначить категорию `'E'`, а клиенту, получающему 235000,
# — `'B'`. Используйте собственную функцию с именем `categorize_income()` и метод `apply()`.**


def categorize_income(income):
    try:
        if 0 <= income <= 30000:
            return 'E'
        elif 30001 <= income <= 50000:
            return 'D'
        elif 50001 <= income <= 200000:
            return 'C'
        elif 200001 <= income <= 1000000:
            return 'B'
        elif income >= 1000001:
            return 'A'
    except:
        pass


data['total_income_category'] = data['total_income'].apply(categorize_income)

# **Задание 17. Выведите на экран перечень уникальных целей взятия кредита из столбца `purpose`.**

print("ex-17\n", data['purpose'].unique())


# **Задание 18. Создайте функцию, которая на основании данных из столбца `purpose` сформирует новый столбец
# `purpose_category`, в который войдут следующие категории:**
# 
# - `'операции с автомобилем'`,
# - `'операции с недвижимостью'`,
# - `'проведение свадьбы'`,
# - `'получение образования'`.
# 
# **Например, если в столбце `purpose` находится подстрока `'на покупку автомобиля'`, то в столбце `purpose_category`
# должна появиться строка `'операции с автомобилем'`.**
# 
# **Используйте собственную функцию с именем `categorize_purpose()` и метод `apply()`. Изучите данные в столбце
# `purpose` и определите, какие подстроки помогут вам правильно определить категорию.**


def categorize_purpose(row):
    try:
        if 'автом' in row:
            return 'операции с автомобилем'
        elif 'жил' in row or 'недвиж' in row:
            return 'операции с недвижимостью'
        elif 'свад' in row:
            return 'проведение свадьбы'
        elif 'образов' in row:
            return 'получение образования'
    except:
        return 'нет категории'


data['purpose_category'] = data['purpose'].apply(categorize_purpose)


# ### Шаг 3. Исследуйте данные и ответьте на вопросы

# #### 3.1 Есть ли зависимость между количеством детей и возвратом кредита в срок?

def children_category(children):
    if 1 <= children <= 2:
        return 'есть дети'
    return 'нет детей'


data['children_rank'] = data['children'].apply(children_category)
data.tail()
debt_children = data.pivot_table(index='children_rank', columns='debt', values='gender', aggfunc='count')
debt_children.columns = ['нет долгов', 'долг']
debt_children['доля должников'] = debt_children['долг'] / (debt_children['долг'] + debt_children['нет долгов'])
debt_children['доля должников'] = debt_children['доля должников'].map('{:.1%}'.format)
print("Вопрос 3.1\n", debt_children)

# **Вывод:** у людей с детьми, больше вероятность стать должником

# #### 3.2 Есть ли зависимость между семейным положением и возвратом кредита в срок?

debt_family_s = data.pivot_table(index='family_status', columns='debt', values='gender', aggfunc='count')
debt_family_s.columns = ['нет долгов', 'долг']
debt_family_s['доля'] = debt_family_s['долг'] / (debt_family_s['долг'] + debt_family_s['нет долгов'])
debt_family_s['доля'] = debt_family_s['доля'].map('{:.1%}'.format)
print("Вопрос 3.2\n", debt_family_s)


# **Вывод:**  те кто не женат имеют больший шанс стать должником

# #### 3.3 Есть ли зависимость между уровнем дохода и возвратом кредита в срок?


def categorize_income(income):
    try:
        if 0 <= income <= 30000:
            return 'E'
        elif 30001 <= income <= 50000:
            return 'D'
        elif 50001 <= income <= 200000:
            return 'C'
        elif 200001 <= income <= 1000000:
            return 'B'
        elif income >= 1000001:
            return 'A'
    except:
        pass


data['total_income_grup'] = data['total_income'].apply(categorize_income)
debt_total_i = data.pivot_table(index='total_income_grup', columns='debt', values='gender', aggfunc='count')
debt_total_i.columns = ['нет долгов', 'долг']
debt_total_i['доля'] = debt_total_i['долг'] / (debt_total_i['долг'] + debt_total_i['нет долгов'])
debt_total_i['доля'] = debt_total_i['доля'].map('{:.1%}'.format)
print("Вопрос 3.3\n", debt_total_i)

# **Вывод:** у людей со средним доходом больше шанс стать должником чем у людей с низким уровнем (А)

# #### 3.4 Как разные цели кредита влияют на его возврат в срок?

debt_purpose_c = data.pivot_table(index='purpose_category', columns='debt', values='gender', aggfunc='count')
debt_purpose_c.columns = ['нет долга', 'долг']
debt_purpose_c['доля'] = debt_purpose_c['долг'] / (debt_purpose_c['долг'] + debt_purpose_c['нет долга'])
debt_purpose_c['доля'] = debt_purpose_c['доля'].map('{:.1%}'.format)
print("Вопрос 3.4\n", debt_purpose_c)

# **Вывод:** люди, берущие кредит на получения образования или покупку автомобиля имеют более высокий риск стать
# должником

# #### 3.5 Приведите возможные причины появления пропусков в исходных данных.

# *Ответ:* ошибки при вводе данных, технические проблемы

# #### 3.6 Объясните, почему заполнить пропуски медианным значением — лучшее решение для количественных переменных.

# *Ответ:* так как могут встречаться аномальные значения

# ### Шаг 4: общий вывод.

# Напишите ваш общий вывод. В зависимости от семейного положения, наличию детей, цели кредита и дохода, банк может
# понять каким людям лучше отказать в кредите, а каким его одобрить. Так же необходимо проверять правильность
# вводимых данных, для избежания ошибок.

# **Вывод:** Люди имеющие самые большие риски на невыплату кредита: не состоящие в браке, люди с детьми, люди берущие
# кредит на получения образования или покупку автомобиля, люди со средним доходом. Таким клиентам лучше отказать в
# кредите. В свою очередь, люди которые чаще выплачивают кредит: вдовцы и вдовы, люди без детей, люди берущие кредит на
# недвижимость, и люди с доходом категории D (от 30001 до 50000). Соответственно, кредит таким клиентам можно одобрять!

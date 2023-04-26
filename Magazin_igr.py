#!/usr/bin/env python
# coding: utf-8

# Нет необходимого вступления в проект: нужно рассказать, что за данные у тебя имеются, что вообще стоит делать,
# какие цели и задачи, что желаешь получить. Представь, что показываешь проект бизнесу / заказчику: заказчик не
# поймет даже, где оказался, поэтому нужно стороннего читателя вводить в курс дела </div>

# Я работаю в интернет-магазине «Игромир», который продаёт по всему миру компьютерные игры. У нас имеется
# база-данных о пользователях игр. Для начала будет необходимо отсортировать все данные приведенные в датасете,
# для комфортной работы, большинство ошибок с данными возникают из-за человеческого фактора, но не все. Основная цель
# нашего исследования - сделать прогноз о том какие игры и какие площадки будут востребованы.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as st

game = pd.read_csv('games.csv')

# 1. Открываем и просматриваем данные

print('ex-1')
game.head(10)
game.info()
game.columns = map(str.lower, game.columns)

game['year_of_release'] = game['year_of_release'].astype('Int64')

# Убираем ноль после точки для обработки данных

game.fillna({"name": " ", "genre": " "}, inplace=True)

game['user_score'] = game['user_score'].replace('tbd', np.NaN)
game['user_score'] = game['user_score'].astype('float')
game['critic_score'] = game['critic_score'].astype('Int64')
game['total_s'] = game['na_sales'] + game['eu_sales'] + game['jp_sales'] + game['other_sales']

# Заменим пропущенные значения на "no"

game["rating"] = game['rating'].fillna("no")
game.head(10)


# 2. Исследовательский анализ данных

# смотрим есть ли необходимость анализировать данные за все время


def create_any_bar(groupby_column, func, y='name'):
    df_to_plot = game.groupby(groupby_column)[y]
    if func == 'count':
        df_to_plot_calculated = df_to_plot.count()
        figsize = (15, 5)
        plot = df_to_plot_calculated.plot(kind='bar', y=y, figsize=figsize, ec='black')

    elif func == 'sum':
        df_to_plot_calculated = df_to_plot.sum().sort_values()
        figsize = (15, 10)
        plot = df_to_plot_calculated.plot(kind='barh', y=y, figsize=figsize, ec='black')
    plt.show()


# create_any_bar('year_of_release', 'count')
# create_any_bar('platform', 'sum', 'total_s')

# Ожидаемый результат - на самых популярных платформах большое кол-во проданных игр

games_1994 = game[game['year_of_release'] >= 1994]

platforms_leaders = games_1994.groupby('platform')['total_s'].sum().sort_values()[-5:]

print('ex-2')
print(platforms_leaders)

# Самые популярные платформы

game = game[game['year_of_release'] >= 2013]
platforms_leaders1 = game.groupby('platform')['total_s'].sum().sort_values()[-3:]
print(platforms_leaders1)

game_top_5 = game[game['platform'].isin(['PS4', 'XOne', 'PC', 'WiiU'])]
game_top_5 = game_top_5[game_top_5['total_s'] < 1.4]
game_top_5.groupby('platform')['total_s'].describe()

game.pivot_table(index='name', columns='platform', values='total_s', aggfunc='sum').plot(kind='box', ylim=(0, 1.9))
# plt.show()

# На графике мы видим, что мы не учли консоль только набирающую популярность - XOne. На нее было выпущено не так
# много игр, поэтому мы ее не включили в предыдущий график, но игры этой консоли очень хорошо продаются, потому что
# медианные значения PS4 и XOne почти совпадают. Добавим XOne в список потенциально прибыльных.

# Чем лучше рейтинг, тем игра лучше, следовательно, больше продаж, но есть и "аномалии" - в столбце user_score есть
# игры, которые имеют высокую оценку пользователей, но при этом мало продались.

print("Матрица корреляций")
game[game.platform == "PS4"][['total_s', 'critic_score', 'user_score']].corr()

distr_genre = game_top_5.pivot_table(index='genre', values='total_s', aggfunc='sum').sort_values(
    by='total_s', ascending=False).reset_index().rename_axis(None, axis=1)
print(distr_genre)

# 3. Посмотрим на общее распределение игр по жанрам. Что можно сказать о самых прибыльных жанрах? Выделяются ли жанры с
# высокими и низкими продажами?

distr_genre.groupby('genre').median()['total_s'].sort_values(ascending=False)
print('ex-3')
print(distr_genre)

# Больше всего игр жанра Action, затем идут Shooter, sports

# Общие продажи/кол-во продаж - плохая метрика для поиска наиболее прибыльного жанра. За высокими показателями общих
# продаж может скрываться множество мелких игр с низкими продажами. Или 2-3 звезды и куча провалов. Лучше найти жанр,
# где игры стабильно приносят высокий доход - для этого стоит рассмотреть медианные продажи. Также общие продажи
# неустойчивы к выбросам, мы сделаем неверные выводы, если будем отталкиваться от этих данных, потому нам нужны
# медианные продажи

ax = plt.gca()

pivot = game.groupby('genre').agg({'name': 'count', 'total_s': 'sum'}).sort_values(by='name', ascending=False)

plot1 = pivot['name']
plot1.plot(kind='bar', figsize=(15, 5), ec='black', ax=ax, width=0.2, position=1)

plot2 = pivot['total_s']
plot2.plot(kind='bar', figsize=(15, 5), ec='black', ax=ax, width=0.2, color='#97F0AA', position=0)

ax.legend(['Количество продаж', 'Общая сумма продаж'])


def top_in_regions_plot(groupby, region_sales, ax):
    pivot = game.groupby(groupby).agg({region_sales: 'sum'}).sort_values(by=region_sales, ascending=False)[:5]
    title_dict = {'na_sales': 'North America Sales', 'eu_sales': 'Europe Sales', 'jp_sales': 'Japan Sales'}
    color_dict = {'na_sales': 'red', 'eu_sales': 'gray', 'jp_sales': 'forestgreen'}
    plot = pivot.plot(kind='bar', ec='black', title=title_dict[region_sales], ax=plt.axes[ax],
                      fontsize=18, color=color_dict[region_sales], rot=20)
    plot.legend(prop={'size': 17})
    plot.set_xlabel('')
    plot.title.set_size(20)


print('Cамые популярные платформы (топ-5)')
fig, axes = plt.subplots(1, 3, figsize=(25, 6))

# top_in_regions_plot('platform', 'na_sales', 0)
# top_in_regions_plot('platform', 'eu_sales', 1)
# top_in_regions_plot('platform', 'jp_sales', 2)

# Больше всего покупают игры жанра Action, возможно, это связано с тем, что игр данного жанра выпускается больше всех
# остальных, сам по себе жанр много в себя вбирает. Sports идёт на втором месте и по количеству выпускаемых игр и по
# продажам.  Shooter неожиданно стоит на третьем месте по продажам, хотя игр производится почти в два раза меньше.
# Неожиданно, что Adventure сильно отстаёт по продажам, хотя игр выпускается много

# Средние пользовательские рейтинги платформ Xbox One и PC одинаковые.
#
# Н0 (нулевая гипотеза) - средние пользовательские рейтинги платформ Xbox One и PC одинаковые.
#
# Н1 (альтернативная гипотеза) - средние пользовательские рейтинги платформ Xbox One и PC отличаются между собой.

# 4. Оценки пользователей Xbox One и PC

sample_1 = game[game['platform'] == "XOne"]['user_score'].dropna()
sample_2 = game[game['platform'] == "PC"]['user_score'].dropna()

from scipy.stats import levene

stat, p = levene(sample_1, sample_2)
print('ex-4')
print(stat, p)

alpha = 0.05

results_1 = st.ttest_ind(
    sample_1,
    sample_2,
    equal_var = True)

print('p-значение:', results_1.pvalue)

if (results_1.pvalue < alpha):
    print("Отвергаем нулевую гипотезу")
else:
    print("Не получилось отвергнуть нулевую гипотезу")

# Не получилось отвергнуть нулевую гипотезу.
#
# Верно следующее утверждение: средние пользовательские рейтинги платформ Xbox One и PC одинаковые,
# средние пользовательские рейтинги жанров Action и Sports разные.
#
# Н0 (нулевая гипотеза) - средние пользовательские рейтинги жанров Action и Sports одинаковые.
# Н1 (альтернативная гипотеза) - средние пользовательские рейтинги жанров Action и Sports отличаются между собой.

sample_3 = game[game['genre'] == "Action"]['user_score'].dropna()
sample_4 = game[game['genre'] == "Sports"]['user_score'].dropna()

alpha = 0.05

results_2 = st.ttest_ind(
    sample_3,
    sample_4,
    equal_var = False)

print('p-значение:', results_2.pvalue)

if (results_2.pvalue < alpha):
    print("Отвергаем нулевую гипотезу")
else:
    print("Не получилось отвергнуть нулевую гипотезу")


# Отвергаем нулевую гипотезу. Таким образом, остается верным утверждение, что средние пользовательские рейтинги
# жанров Action и Sports отличаются.
# Для проверки обеих гипотез будем использовать t-критерий Стьюдента, т.к. в обоих случаях выборки независимы между
# собой.

# Вывод
#
#  Изучение и подготовка данных
# Ознакомились с данными, посмотрели на общую информацию о датасете, выявили пропуски и не соответствие типов данных.
# На данном шаге мы заполнили некоторые пропуски
# Многие пропуски, например в столбцах оценок мы оставили незаполненными, чтобы не исказить статистику.
# Также на данном этапе мы посчитали общее количество продаж по всем регионам и записали результат в столбец total_s.
#
# Исследовательский анализ данных
# Было обнаружено, что рост выпуска игр приходится на 1994 год, а пик на 2008-2009 гг.
# Характерный срок жизни платформы - 10 лет, поэтому оставим данные с 2013 по 2016 гг.
# 3 потенциально прибыльных платформ - PS4, XOne, 3DS.
# Наибольшие медианные продажи у платформ X360 и PS3.
# Почти у всех платформ есть определенные игры, которые стали популярными, в большинстве своем это эксклюзивы
# Компьютерные игры PC стоят дешевле консольных.
# Больше всего игр жанра Action
# Лучше всего покупают игры жанра Action
# Adventure сильно отстаёт по продажам, хотя игр выпускается много.
#
# Портрет пользователя региона В NA самая популярная платформа X360. Европейцы предпочитают PS3. В Японии популярны
# DS. В NA и EU самые популярные жанры практически совпадают. В JP вкусы отличаются. Во всех регионах лидируют игры с
# рейтингом E - "Для всех". В Европе и Северной Амереке дальше идут по "старшенству". В Японии опять не так. Первое
# место такое же - "Для всех", а вот на втором - игры для лиц от 13 лет, далее 17+. Проверка гипотез Средние
# пользовательские рейтинги платформ Xbox One и PC одинаковые. Гипотеза не подтвердилась. Средние пользовательские
# рейтинги жанров Action и Sports разные. Гипотеза подтвердилась!.

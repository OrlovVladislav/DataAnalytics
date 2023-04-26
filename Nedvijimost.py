#!/usr/bin/env python
# coding: utf-8

# # Исследование объявлений о продаже квартир
#
# В вашем распоряжении данные сервиса Яндекс.Недвижимость — архив объявлений о продаже квартир в Санкт-Петербурге и
# соседних населённых пунктов за несколько лет. Нужно научиться определять рыночную стоимость объектов недвижимости.
# Ваша задача — установить параметры. Это позволит построить автоматизированную систему: она отследит аномалии и
# мошенническую деятельность. По каждой квартире на продажу доступны два вида данных. Первые вписаны пользователем,
# вторые — получены автоматически на основе картографических данных. Например, расстояние до центра, аэропорта,
# ближайшего парка и водоёма.

# ### 1. Откройте файл с данными и изучите общую информацию.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('real_estate_data.csv', sep='\t')

print('ex-1')
data.info()

# ### 2. Предобработка данных

data.rename(columns={'cityCenters_nearest': 'city_centers_nearest'}, inplace=True)
data.isna().sum()

data['is_apartment'] = data['is_apartment'].fillna('False')
data['balcony'] = data['balcony'].fillna(0)
data['locality_name'].dropna(inplace=True)
data['floors_total'].dropna(inplace=True)

print('ex-2')

print('Квартир с потолками от 4 до 5м: ' + str(
    data.query('ceiling_height > 4 and ceiling_height <= 5')['ceiling_height'].count()) + ' шт.')
print('Квартир с потолками от 5 до 25м: ' + str(
    data.query('ceiling_height > 5 and ceiling_height < 25')['ceiling_height'].count()) + ' шт.')
print('Квартир с потолками от 25м до 36м: ' + str(
    data.query('ceiling_height >= 25 and ceiling_height <= 36')['ceiling_height'].count()) + ' шт.')
print('Квартир с потолками от 25м до 36м: ' + str(
    data.query('ceiling_height > 36 and ceiling_height < 50')['ceiling_height'].count()) + ' шт.')
print('Квартир с потолками от 50м: ' + str(data.query('ceiling_height >= 50')['ceiling_height'].count()) + ' шт.')

data.drop(index=data.query('ceiling_height > 4 and ceiling_height < 25').index, inplace=True)
data.update(data[(data['ceiling_height'] >= 25) & (data['ceiling_height'] <= 36)]['ceiling_height'] / 10)

data.drop(index=data.query('ceiling_height > 36').index, inplace=True)
print('Квартир с потолками меньше 2.5м : ' + str(data.query('ceiling_height < 2.5')['ceiling_height'].count()) + ' шт.')
# data['ceiling_height'] = data['ceiling_height'].fillna(data.mean())
data['ceiling_height'].dropna(inplace=True)
data.info()

# ### 3. Сменить тип данных для более удобной работы с данными, ошибки скорее всего возникили из-за человеческого
# фактора

print('ex-3')

try:
    data['last_price'] = data['last_price'].astype('int64')
    print('1ok')
except:
    print('last_price int type conversion fail')

try:
    data['first_day_exposition'] = pd.to_datetime(data['first_day_exposition'], format='%Y-%m-%dT%H:%M:%S')
    print('2ok')
except:
    print('first_day_exposition to_datetime type conversion fail')

try:
    data['floors_total'] = data['floors_total'].astype('Int8')
    print('3ok')
except:
    print('floors_total int8 type conversion fail')

try:
    data['is_apartment'] = data['is_apartment'].map({'False': False, 'True': True})
    print('4ok')
except:
    print('is_apartment bool type conversion fail')

try:
    data['balcony'] = data['balcony'].astype('int8')
    print('5ok')
except:
    print('balcony int8 type conversion fail')

try:
    data['days_exposition'] = np.floor(pd.to_numeric(data['days_exposition'], errors='coerce')).astype('Int64')
    print('6ok')
except:
    print('days_exposition int64 type conversion fail')

data.info()

# ### 4. Удаление дубликатов

print('ex-4')

print('Количество уникальных занчений locality_name:', data['locality_name'].unique().shape[0])

data['locality_name'] = data['locality_name'].str.replace('ё', 'е')
data['locality_name'] = data['locality_name'].str.replace('городской поселок', 'поселок городского типа')

print('Количетво явных дубликатов:', data.duplicated().sum())

# ### 5. Выбросы

print('ex-5')

data.describe()
print('Квартир стоимостью больше 100.000.000:', data.query('last_price > 1e+08')['last_price'].count())
print('Квартиры без комнат:', data.query('rooms == 0')['rooms'].count())

data.drop(index=data.query('ceiling_height < 2.5').index, inplace=True)
data.drop(index=data.query('living_area < 10').index, inplace=True)
data.drop(index=data.query('living_area > 200').index, inplace=True)
data.drop(index=data.query('kitchen_area < 2').index, inplace=True)
data.drop(index=data.query('kitchen_area > 50').index, inplace=True)
print('Объявлению больше 3 лет : ', data.query('days_exposition > 365*3')['days_exposition'].count())

data.drop(index=data.query('days_exposition > 365*3').index, inplace=True)
print('Квартир слишком близко к аэропорту:', data.query('airports_nearest < 5000')['airports_nearest'].count())
data.drop(index=data.query('airports_nearest < 5000').index, inplace=True)

data.reset_index(drop=True, inplace=True)

# (
#     data[{'rooms', 'total_area', 'ceiling_height', 'days_exposition', 'last_price', 'living_area', 'kitchen_area',
#           'floor',
#           'floors_total'}]
#     .apply(['count', 'min', 'max'])
#     .style.format("{:,.2f}")
# )

data.rooms.value_counts().to_frame()

# ### 6. Посчитайте и добавьте в таблицу новые столбцы

print('ex-6')

# Цена 1 кв метра

data['cost_per_sqm'] = data['last_price'] / data['total_area']

# Месяц публикации

data['dayofweek'] = data['first_day_exposition'].dt.dayofweek
data['month'] = data['first_day_exposition'].dt.month
data['year'] = data['first_day_exposition'].dt.year

# Расстояние до центра

data['floor_category'] = data.apply(
    lambda x: 'первый' if (x.floor == 1)
    else ('последний' if (x.floor == x.floors_total) & isinstance(x.floors_total, int)
          else 'другой'), axis=1
)

data['floor_category_digit'] = data.apply(
    lambda x: 0 if (x.floor == 1)
    else (2 if (x.floor == x.floors_total) & isinstance(x.floors_total, int)
          else 1), axis=1
)

data['city_centers_nearest_km'] = round(data['city_centers_nearest'] / 1000)

try:
    data['city_centers_nearest_km'] = data['city_centers_nearest_km'].astype('Int32')
    print('city_centers_nearest_km ok')
except:
    print('city_centers_nearest_km Int32 type conversion fail')

data.info()

# ### 7. Проведите исследовательский анализ данных

print('ex-7')

data.describe()
data.hist('total_area', bins=100)
# plt.show()

# общая площадь; - Наблюдаем очень малое количество квартир с общей площадью более 100 кв.м.

data.hist('living_area', bins=100)
# plt.show()

# жилая площадь; - Наблюдаем два пика - на 18 кв.м. и на 30 кв.м. Нужно проверить, почему у нас именно два пика и
# почему есть явный провал около 24 кв.м.

data.hist('kitchen_area', bins=100)
# plt.show()

# площадь кухни; - обычные показатели, больше зависит от общей площади квартиры

data.hist('last_price', bins=100, range=(0, 2e+07))
# plt.show()

# в правой части наблюдается небольшой хвост, скорее всего это элитная недвижимость, но остальной график выглядит хорошо

data.hist('rooms', bins=data['rooms'].max())
# plt.show()

# количество комнат;  - Больше всего двух- и трёх-комнатных квартир, но встречаются и редкие исключения.

data.hist('ceiling_height', bins=30)
# plt.show()

# высота потолков; - Здесь два пика 2.5м и 3м, что логично.

data.groupby(by='floor_category')['floor_category'].count().plot(kind='bar', ylabel='count')
# plt.show()

# этаж квартиры; - больше всего квартир на "другом" этаже и это логично.

data.hist('floors_total', bins=data['floors_total'].max())
# plt.show()

# общее количество этажей в доме; - Больше всего квартир с 1 по 5 этаж.

data.hist('city_centers_nearest', bins=100)
# plt.show()

# Расстояние до центра города в метрах; - Здесь наблюдаем два пика, один маленький, 5км и другой побльшой от 10 км.
# до 15 км. Так же есть совсем небольшие всплески на 30 км. и 50 км.

data.hist('airports_nearest', bins=100)
# plt.show()

# Расстояние до ближайшего аэропорта; - Здесь видим довольно "шумный" график. Но это вполне нормально.

data.hist('parks_nearest', bins=100)
# plt.show()

# расстояние до ближайшего парка; - Судя по графику больше всего квартир с парками на расстоянии до 750м.

data.hist('dayofweek', bins=7)
# plt.show()

# день и месяц публикации объявления. - Видим провал в публикации объявлений в субботу и воскресенье, что логично.

data.hist('month', bins=12)
# plt.show()

# с февраля по апрель наблюдалосась повышенная активность. Так же был плавный рост активности с июля по ноябрь.
# Провалы в декабре, январе и мае.

(
    data['ceiling_height']
    .sort_values()
    .plot(y='ceiling_height', kind='hist',
          bins=300, range=(1.9, 4.5), grid=True, title='Высота потолков', figsize=(16, 5))
    .set(ylim=(0, 300), ylabel='кол-во записей', xlabel='высота, м.')

)
# plt.show()

data.hist('days_exposition', bins=100)
data.hist('days_exposition', bins=100, range=(0, 100))
print(f'Среднее время продажи квартиры в днях:', int(data['days_exposition'].mean()))
print('Медианное время продажи квартиры в днях:', int(data['days_exposition'].median()))
print('\n[Выбросы] Количество объявлений, которые сняты через:')
print('45 дней:', data[data['days_exposition'] == 45]['days_exposition'].count())
print('60 дней:', data[data['days_exposition'] == 60]['days_exposition'].count())
print('90 дней:', data[data['days_exposition'] == 90]['days_exposition'].count())

# Можно сказать, что среднее время продажи квартиры составляет 181 день или целые полгода. Но если взять медиану - то
# это уже 95 дней, в два раза меньше. Почему так? Потому что у нас есть "длинный хвост" квартир, которые продавались
# очень долго, буквально годами. Я бы предложил считать быстрыми продажи до 95 дней, а необычно долгими - свыше 181
# дня. Выбросы похожи на платные объявления с истекшим сроком размещения или работу системы удаления неактивных
# объявлений.


# data.plot(x='last_price', y='living_area', kind='scatter', alpha=0.2)
# data.plot(x='last_price', y='kitchen_area', kind='scatter', alpha=0.2)
# data.pivot_table(index='rooms', values='last_price').plot(y='last_price', kind='bar')
# data.pivot_table(index='dayofweek', values='last_price', aggfunc='mean').plot(y='last_price', kind='line',
#                                                                               title='mean')
# data.pivot_table(index='dayofweek', values='last_price', aggfunc='median').plot(y='last_price', kind='line',
#                                                                                 title='median')
# data.pivot_table(index='month', values='last_price', aggfunc='mean').plot(y='last_price', kind='line', title='mean')
# data.pivot_table(index='month', values='last_price', aggfunc='median').plot(y='last_price', kind='line',
#                                                                             title='median')
# data.pivot_table(index='year', values='last_price', aggfunc='mean').plot(y='last_price', kind='line', title='mean')
# data.pivot_table(index='year', values='last_price', aggfunc='median').plot(y='last_price', kind='line', title='median')
# data.pivot_table(index='floor_category', values='last_price').plot(y='last_price', kind='barh')
# data.plot(x='last_price', y='floor_category', kind='scatter', alpha=0.2)

# Цена зависит от:
#
# общей площади;
# жилой площади;
# площади кухни;
# количества комнат.

# Чем больше показатель этих значений тем выше будет цена.
#
# Цена практически не зависит от:
#
# этажа, на котором расположена квартира;
# даты размещения.

top_10 = data.pivot_table(index='locality_name', values=['last_price', 'total_area'], aggfunc=['sum', 'count'])
top_10.columns = ['last_price_sum', 'total_area_sum', 'last_price_count', 'total_area_count']
top_10.pop('total_area_count')
top_10.sort_values(by=['last_price_count'], ascending=False, inplace=True)
top_10 = top_10.iloc[:10]
top_10['price_per_sq_m'] = top_10['last_price_sum'] / top_10['total_area_sum']
top_10.sort_values(by=['price_per_sq_m'], ascending=True, inplace=True)
top_10['price_per_sq_m'].plot(kind='barh')
data.groupby(by='rooms')['rooms'].count().sort_values(ascending=False)
print(data['rooms'].value_counts())

data['rooms'].value_counts()
data.groupby(by='rooms')['rooms'].count().sort_values(ascending=False)

(
    data[data['locality_name'] == 'Санкт-Петербург']
    .pivot_table(
        index='city_centers_nearest_km',
        values='last_price',
        aggfunc='mean')
    .plot(kind='bar')
)

# С большим отрывом лидирует недвижимость до 1км.
# Видим более низкие цены на недвижимость на расстоянии от 1 км. до 7 км. включительно.
# Далее цена спадает на расстоянии от 8 км. до 27 км.
# Видим пик на 27км, его сложно объяснить, возможно это недвижимость в "особом" районе.

# ### Общий вывод

# При работе в дата фреймом были выполнены различные действия для более точного анализа данных, например были
# удалены дубликаты, исправлены неправильные значения, заменены пропуски, добавлены новые столбцы,после того как
# данные стали пригодными для анализаЮ были построены графики отображающие различные параметры. На основе
# проделанной работы, можно сделать выводы: Ошибки в данных могут возникнуть из-за различных факторов,
# один из основных это человеческий общая площадь; - Наблюдаем очень малое количество квартир с общей площадью более
# 100 кв.м. жилая площадь; - Наблюдаем два пика - на 18 кв.м. и на 30 кв.м. площадь кухни; - обычные показатели,
# больше зависит от общей площади квартиры Итоговая цена-можем наблюдать среднюю стоимость квартиры,
# но также присутствуют очень дорогие квартиры количество комнат;  - Больше всего двух- и трёх-комнатных квартир,
# но встречаются и редкие исключения. высота потолков; - Здесь два пика 2.5м и 3м этаж квартиры; - на первом и
# последнем этаже примерно одинаковое количество квартир, на других этажах показатель гораздо выше общее количество
# этажей в доме; - Больше всего квартир с 1 по 5 этаж. Расстояние до центра города в метрах; - Здесь наблюдаем два
# пика, один маленький, 5км и другой побольше от 10 км. до 15 км. Так же есть совсем небольшие всплески на 30 км. и
# 50 км. расстояние до ближайшего парка; - Судя по графику больше всего квартир с парками на расстоянии до 750м. цена
# недвижимости зависит от:
#
# общей площади; жилой площади; площади кухни; количества комнат.
#
# Чем больше показатель этих значений тем выше будет цена.
#
# Цена практически не зависит от:
#
# этажа, на котором расположена квартира; даты размещения.
#
# Цена квартиры чаще всего зависит от ее площади, но так же влияют факторы того, где находится эта квартира,
# например квартире ближе к центру будет стоить дороже, чем квартира на отшибе. Рядом с таким крупным городом
# располагаются и другие города в которых также продаются квартиры, но онм не такие дорогие как в Санкт-Петербурге
# Срок продажи квартир зависит от многих параметров, в результате анализа было выявлено, что медианное значение равно
# - 95 дням, а среднее 171

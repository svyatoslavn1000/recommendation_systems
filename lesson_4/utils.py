import numpy as np
import pandas as pd


def prefilter_items(data, take_n_popular=None):
    # Уберем самые популярные товары (их и так купят)
    popularity = (data.groupby('item_id')['user_id'].nunique() / data['user_id'].nunique()).reset_index()
    popularity.rename(columns={'user_id': 'share_unique_users'}, inplace=True)    
    
    top_popular = popularity[popularity['share_unique_users'] > 0.5]['item_id'].unique()
    data = data[~data['item_id'].isin(top_popular)]
    
    # Уберем самые НЕ популярные товары (их и так НЕ купят)
    top_notpopular = popularity[popularity['share_unique_users'] < 0.01]['item_id'].unique()
    data = data[~data['item_id'].isin(top_notpopular)]
    
    # Уберем товары, которые не продавались за последние 12 месяцев
    last_week_no = np.sort(data['week_no'].unique())[-1]
    sold_this_year = data[data['week_no'] > (last_week_no - 52)]['item_id'].unique()
    data = data[data['item_id'].isin(sold_this_year)]
    
    # Уберем слишком дешевые товары (на них не заработаем). 1 покупка из рассылок стоит 60 руб. 
    prices = data[['item_id']]
    prices['price'] = data['sales_value'] / data['quantity']
    prices = prices.groupby('item_id')['price'].mean().reset_index()
    
    cheap_items = prices.loc[prices['price'] < prices['price'].quantile(0.05)]['item_id'].unique()
    data = data[~data['item_id'].isin(cheap_items)]
    
    # Уберем слишком дорогие товарыs
    expensive_items = prices.loc[prices['price'] > prices['price'].quantile(0.95)]['item_id'].unique()
    data = data[~data['item_id'].isin(expensive_items)]
    
    # Возьмем только топ 5000 популярных товаров из оставшихся
    if take_n_popular is not None:
        items_sold = data.groupby('item_id')['quantity'].sum().reset_index()
        top_5000 = items_sold.sort_values('quantity', ascending=False).iloc[:take_n_popular]['item_id'].unique()
        data.loc[~data['item_id'].isin(top_5000), 'item_id'] = 999999
    
    return data
    
    
def postfilter_items(user_id, recommednations):
    pass

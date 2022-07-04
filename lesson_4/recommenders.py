import pandas as pd
import numpy as np

# Для работы с матрицами
from scipy.sparse import csr_matrix

# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender  # нужен для одного трюка
from implicit.nearest_neighbours import bm25_weight, tfidf_weight


class MainRecommender:
    """Рекоммендации, которые можно получить из ALS
    
    Input
    -----
    user_item_matrix: pd.DataFrame
        Матрица взаимодействий user-item
    """
    
    def __init__(self, data, weighting=True):
        # your_code. Это не обязательная часть. Но если вам удобно что-либо посчитать тут - можно это сделать
        self.user_item_matrix = self.prepare_matrix(data)  # pd.DataFrame
        self.id_to_itemid, self.id_to_userid, \
            self.itemid_to_id, self.userid_to_id = self.prepare_dicts(self.user_item_matrix)
        
        if weighting:
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T
        
        self.model = self.fit(self.user_item_matrix)
        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)
     
    @staticmethod
    def prepare_matrix(data):
        # your_code
        data = MainRecommender.prefilter_items(data, take_n_popular=5000)
        user_item_matrix = pd.pivot_table(data,
                                          index='user_id',
                                          columns='item_id',
                                          values='quantity',
                                          aggfunc='count',
                                          fill_value=0)
        user_item_matrix = user_item_matrix.astype(float)
        
        return user_item_matrix
    
    @staticmethod
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
    
    @staticmethod
    def prepare_dicts(user_item_matrix):
        """Подготавливает вспомогательные словари"""
        
        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))

        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))
        
        return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id
     
    @staticmethod
    def fit_own_recommender(user_item_matrix):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""
    
        own_recommender = ItemItemRecommender(K=1, num_threads=4)
        own_recommender.fit(csr_matrix(user_item_matrix).T.tocsr())
        
        return own_recommender
    
    @staticmethod
    def fit(user_item_matrix, factors=20, regularization=0.001, iterations=15, num_threads=4):
        """Обучает ALS"""
        
        model = AlternatingLeastSquares(factors=factors, 
                                             regularization=regularization,
                                             iterations=iterations,  
                                             num_threads=num_threads)
        model.fit(csr_matrix(user_item_matrix).T.tocsr())
        
        return model

    def get_similar_items_recommendation(self, user, N=5):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""

        # your_code
        # Практически полностью реализовали на прошлом вебинаре
        # Берем топ-N покупок юзера, для каждой из них рекомендуем ближайший (самый похожий) товар
        
        top_n_items = pd.DataFrame(self.user_item_matrix.toarray()).loc[self.userid_to_id[user]]\
                                                                   .sort_values(ascending=False)[:N].index.tolist()
        res = [id_to_itemid[self.model.similar_items(item, N=2)[1][0]] for item in top_n_items]
        
        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res
    
    def get_similar_users_recommendation(self, user, N=5):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""
    
        # your_code
        # Находим топ-N наиболее похожих пользователей
        # От каждого пользователя берем их топ товар и рекомендуем
        n_similar_users = 5
    
        similar_users = [self.id_to_userid[row_id] 
                         for row_id, _ in self.model.similar_users(self.userid_to_id[user], N=n_similar_users+1)][1:]

        top_items = pd.Series()
        for sim_user in similar_users:
            top_n_items = pd.DataFrame(self.user_item_matrix.toarray()).loc[self.userid_to_id[sim_user]]\
                                                                       .sort_values(ascending=False)[:N]
            top_items = top_items.append(top_n_items)

        # Усредняем дубликаты
        top_items = top_items.reset_index().groupby('index')[0].mean()

        res = top_items.sort_values(ascending=False)[:5].index.tolist()

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res

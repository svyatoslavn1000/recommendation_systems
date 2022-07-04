import numpy as np


def hit_rate(recommended_list, bought_list):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    flags = np.isin(bought_list, recommended_list)
    return int(flags.sum() > 0)


def hit_rate_at_k(recommended_list, bought_list, k=5):
    return hit_rate(recommended_list[:k], bought_list)


def precision(recommended_list, bought_list):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    flags = np.isin(bought_list, recommended_list)
    return flags.sum() / len(recommended_list)


def precision_at_k(recommended_list, bought_list, k=5):
    return precision(recommended_list[:k], bought_list)


def recall(recommended_list, bought_list):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    flags = np.isin(bought_list, recommended_list)
    return flags.sum() / len(bought_list)


def recall_at_k(recommended_list, bought_list, k=5):
    return recall(recommended_list[:k], bought_list)


def money_recall_at_k(recommended_list, bought_list, prices_recommended, prices_bought, k=5):
    bought_list = np.array(bought_list)
    prices_bought = np.array(prices_bought)
    recommended_list = np.array(recommended_list)[:k]
    
    flags = np.isin(bought_list, recommended_list)
    return np.dot(flags, prices_bought) / prices_bought.sum()


def ap_k(recommended_list, bought_list, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)[:k]
    
    # np.nonzero() - get indexes of nonzero elements of array, returns a tuple with index array as 0th member
    rec_and_relev_idx = np.nonzero(np.isin(recommended_list, bought_list))[0]
    rec_and_relev_count = len(rec_and_relev_idx)
    if rec_and_relev_count == 0:
        return 0
    
    precision_sum = sum([precision_at_k(recommended_list, bought_list, k=idx+1) for idx in rec_and_relev_idx])
    return precision_sum / rec_and_relev_count


def map_k(recommended_list, bought_list, k=5):
    ap_k_sum = sum([ap_k(user_rec_list, user_bought_list, k=k)
                    for user_rec_list, user_bought_list in zip(recommended_list, bought_list)])
    return ap_k_sum / len(recommended_list)


def dcg_discount(item_rank):
    return item_rank if item_rank <= 2 else np.log2(item_rank)


def dcg_k(recommended_list, bought_list, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)[:k]
    
    rec_bought_fact = np.isin(recommended_list, bought_list).astype(int)
    # Ранг начинается с 1
    disc_ranks = np.array([dcg_discount(rank) for rank in range(1, recommended_list.shape[0]+1)])
    
    return (rec_bought_fact / disc_ranks).sum() / recommended_list.shape[0]


def ndcg_k(recommended_list, bought_list, k=5):
    dcg = dcg_k(recommended_list, bought_list, k=k)
    # Сделав так, получим все '1' в bought_fact
    ideal_dcg = dcg_k(recommended_list, recommended_list, k=k)
    
    return dcg / ideal_dcg


def reciprocal_rank_k(recommended_list, bought_list, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)[:k]
    
    rec_bought_fact = np.isin(recommended_list, bought_list).astype(int)
    rec_and_relev_idx = np.nonzero(rec_bought_fact)[0]
    rec_and_relev_count = len(rec_and_relev_idx)
    if rec_and_relev_count == 0:
        return 0
    
    # Ранг начинается с 1, а индексы с 0
    return 1 / (rec_and_relev_idx[0] + 1)


def mrr_k(recommended_list, bought_list, k=5):
    rr_k_sum = sum([reciprocal_rank_k(user_rec_list, user_bought_list, k=k)
                   for user_rec_list, user_bought_list in zip(recommended_list, bought_list)])
    return rr_k_sum / len(recommended_list)

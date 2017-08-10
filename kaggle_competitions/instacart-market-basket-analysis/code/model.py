import numpy as np
import pandas as pd
import xgboost
import gc
from sklearn.model_selection import train_test_split


order_fpath = "../data/orders.csv"
op_prior_fpath = "../data/order_products__prior.csv"
op_train_fpath = "../data/order_products__train.csv"
product_fpath = "../data/products.csv"
aisle_fpath = "../data/aisles.csv"
department_fpath = "../data/departments.csv"


def load_data():
    # order product
    prior = pd.read_csv(op_prior_fpath, dtype={
        'order_id': np.int32, 'product_id': np.uint16, 'add_to_cart_order': np.int16, 'reordered': np.int8})
    train = pd.read_csv(op_train_fpath, dtype={
        'order_id': np.int32, 'product_id': np.uint16, 'add_to_cart_order': np.int16, 'reordered': np.int8})

    # order
    orders = pd.read_csv(order_fpath, dtype={
        'order_id': np.int32, 'user_id': np.int64, 'eval_set': 'category', 'order_number': np.int16,
        'order_dow': np.int8, 'order_hour_of_day': np.int8, 'days_since_prior_order': np.float32})

    # product related
    products = pd.read_csv(product_fpath)
    aisles = pd.read_csv(aisle_fpath)
    departments = pd.read_csv(department_fpath)

    sample_submission = pd.read_csv("../data/sample_submission.csv")

    return prior, train, orders, products, aisles, departments, sample_submission


def ka_add_groupby_features_1_vs_n(df, group_columns_list, agg_dict, only_new_feature=True):
    '''Create statistical columns, group by [N columns] and compute stats on [N column]

       Parameters
       ----------
       df: pandas dataframe
          Features matrix
       group_columns_list: list_like
          List of columns you want to group with, could be multiple columns
       agg_dict: python dictionary

       Return
       ------
       new pandas dataframe with original columns and new added columns
    '''
    try:
        if type(group_columns_list) == list:
            pass
        else:
            raise TypeError(k + "should be a list")
    except TypeError as e:
        print(e)
        raise

    df_new = df.copy()
    grouped = df_new.groupby(group_columns_list)

    the_stats = grouped.agg(agg_dict)
    the_stats.columns = the_stats.columns.droplevel(0)
    the_stats.reset_index(inplace=True)
    if only_new_feature:
        df_new = the_stats
    else:
        df_new = pd.merge(left=df_new, right=the_stats, on=group_columns_list, how='left')

    return df_new


def ka_add_groupby_features_n_vs_1(df, group_columns_list, target_columns_list, methods_list, keep_only_stats=True, verbose=1):
    '''Create statistical columns, group by [N columns] and compute stats on [1 column]

       Parameters
       ----------
       df: pandas dataframe
          Features matrix
       group_columns_list: list_like
          List of columns you want to group with, could be multiple columns
       target_columns_list: list_like
          column you want to compute stats, need to be a list with only one element
       methods_list: list_like
          methods that you want to use, all methods that supported by groupby in Pandas

       Return
       ------
       new pandas dataframe with original columns and new added columns
    '''
    dicts = {"group_columns_list": group_columns_list , "target_columns_list": target_columns_list, "methods_list" :methods_list}

    for k, v in dicts.items():
        try:
            if type(v) == list:
                pass
            else:
                raise TypeError(k + "should be a list")
        except TypeError as e:
            print(e)
            raise

    grouped_name = ''.join(group_columns_list)
    target_name = ''.join(target_columns_list)
    combine_name = [[grouped_name] + [method_name] + [target_name] for method_name in methods_list]

    df_new = df.copy()
    grouped = df_new.groupby(group_columns_list)

    the_stats = grouped[target_name].agg(methods_list).reset_index()
    the_stats.columns = [grouped_name] + ['_%s_%s_by_%s' % (grouped_name, method_name, target_name) for (grouped_name, method_name, target_name) in combine_name]
    if keep_only_stats:
        return the_stats
    else:
        df_new = pd.merge(left=df_new, right=the_stats, on=group_columns_list, how='left')
    return df_new


def create_features(priors, train, orders, products, aisles, departments):
    priors_orders_detail = orders.merge(right=priors, how='inner', on='order_id')
    priors_orders_detail['_user_buy_product_times'] = priors_orders_detail.groupby(['user_id', 'product_id']).cumcount() + 1

    # product
    agg_dict = {'user_id': {'_prod_tot_cnts': 'count'},
                'reordered': {'_prod_reorder_tot_cnts': 'sum'},
                '_user_buy_product_times': {'_prod_buy_first_time_total_cnt': lambda x: sum(x == 1),
                                            '_prod_buy_second_time_total_cnt': lambda x: sum(x == 2)}}
    prd = ka_add_groupby_features_1_vs_n(priors_orders_detail, ['product_id'], agg_dict)
    prd['_prod_reorder_prob'] = prd._prod_buy_second_time_total_cnt / prd._prod_buy_first_time_total_cnt
    prd['_prod_reorder_ratio'] = prd._prod_reorder_tot_cnts / prd._prod_tot_cnts
    prd['_prod_reorder_times'] = 1 + prd._prod_reorder_tot_cnts / prd._prod_buy_first_time_total_cnt

    # user
    agg_dict_2 = {'order_number': {'_user_total_orders': 'max'},
                  'days_since_prior_order': {'_user_sum_days_since_prior_order': 'sum',
                                             '_user_mean_days_since_prior_order': 'mean'}}
    users = ka_add_groupby_features_1_vs_n(orders[orders.eval_set == 'prior'], ['user_id'], agg_dict_2)

    agg_dict_3 = {'reordered':
                      {'_user_reorder_ratio':
                           lambda x: sum(priors_orders_detail.ix[x.index, 'reordered'] == 1) /
                                     sum(priors_orders_detail.ix[x.index, 'order_number'] > 1)},
                  'product_id': {'_user_total_products': 'count',
                                 '_user_distinct_products': lambda x: x.nunique()}}
    us = ka_add_groupby_features_1_vs_n(priors_orders_detail, ['user_id'], agg_dict_3)
    users = users.merge(us, how='inner')

    users['_user_average_basket'] = users._user_total_products / users._user_total_orders

    us = orders[orders.eval_set != "prior"][['user_id', 'order_id', 'eval_set', 'days_since_prior_order']]
    us.rename(index=str, columns={'days_since_prior_order': 'time_since_last_order'}, inplace=True)

    users = users.merge(us, how='inner')

    # user - product
    agg_dict_4 = {'order_number': {'_up_order_count': 'count',
                                   '_up_first_order_number': 'min',
                                   '_up_last_order_number': 'max'},
                  'add_to_cart_order': {'_up_average_cart_position': 'mean'}}

    data = ka_add_groupby_features_1_vs_n(df=priors_orders_detail,
                                          group_columns_list=['user_id', 'product_id'],
                                          agg_dict=agg_dict_4)

    data = data.merge(prd, how='inner', on='product_id').merge(users, how='inner', on='user_id')

    data['_up_order_rate'] = data._up_order_count / data._user_total_orders
    data['_up_order_since_last_order'] = data._user_total_orders - data._up_last_order_number
    data['_up_order_rate_since_first_order'] = data._up_order_count / (data._user_total_orders - data._up_first_order_number + 1)

    train = train.merge(right=orders[['order_id', 'user_id']], how='left', on='order_id')
    data = data.merge(train[['user_id', 'product_id', 'reordered']], on=['user_id', 'product_id'], how='left')

    del orders
    gc.collect()

    return data


def model(data, sample_submission):
    train = data.loc[data.eval_set == "train", :]
    train.drop(['eval_set', 'user_id', 'product_id', 'order_id'], axis=1, inplace=True)
    train.loc[:, 'reordered'] = train.reordered.fillna(0)

    X_test = data.loc[data.eval_set == "test", :]

    X_train, X_val, y_train, y_val = train_test_split(train.drop('reordered', axis=1), train.reordered,
                                                      test_size=0.9, random_state=42)
    d_train = xgboost.DMatrix(X_train, y_train)
    xgb_params = {
        "objective": "reg:logistic"
        , "eval_metric": "logloss"
        , "eta": 0.1
        , "max_depth": 6
        , "min_child_weight": 10
        , "gamma": 0.70
        , "subsample": 0.76
        , "colsample_bytree": 0.95
        , "alpha": 2e-05
        , "lambda": 10
    }

    watchlist = [(d_train, "train")]
    bst = xgboost.train(params=xgb_params, dtrain=d_train, num_boost_round=80, evals=watchlist, verbose_eval=10)
    xgboost.plot_importance(bst)

    d_test = xgboost.DMatrix(X_test.drop(['eval_set', 'user_id', 'order_id', 'reordered', 'product_id'], axis=1))
    X_test.loc[:, 'reordered'] = (bst.predict(d_test) > 0.21).astype(int)
    X_test.loc[:, 'product_id'] = X_test.product_id.astype(str)
    submit = ka_add_groupby_features_n_vs_1(X_test[X_test.reordered == 1],
                                            group_columns_list=['order_id'],
                                            target_columns_list=['product_id'],
                                            methods_list=[lambda x: ' '.join(set(x))], keep_only_stats=True)
    submit.columns = sample_submission.columns.tolist()
    result = sample_submission[['order_id']].merge(submit, how='left').fillna('None')
    result.to_csv("../result/res.csv", index=False)


def main():
    priors, train, orders, products, aisles, departments, sample_submission = load_data()

    data = create_features(priors, train, orders, products, aisles, departments)

    model(data, sample_submission)


if __name__ == "__main__":
    main()


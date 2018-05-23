# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 15:31:32 2017

@author: STP
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import math
import copy
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import pyflux as pf
from polylearn import FactorizationMachineRegressor


def calc_circle(day, circle_min, circle_max, data_new, p):
    compare_dataframe = pd.DataFrame(np.zeros((circle_max + 1 - circle_min, 4), dtype=float), columns=['a', 'b', 'c', 'i'])
    for i in xrange(circle_min, circle_max + 1):
        up = 0
        down = 0
        precent = 0
        for j in xrange(int(len(data_new) / circle_max + 1)):
            start_time2_index = len(day) - 1 - (j + 1) * i
            end_time2_index = len(day) - j * i
            data_bodong = data_new.ix[(start_time2_index + 1):end_time2_index, p]
            up += int(data_bodong.max() - data_bodong.min())
            down += int(abs(data_bodong.head(1).values - data_bodong.tail(1).values))
            precent += float((data_bodong.max() - data_bodong.min()) / float((sum(data_bodong) / i)))
        compare_dataframe.ix[i - 3, 0] = float(up) / float(i)
        compare_dataframe.ix[i - 3, 1] = float(down) / float(i)
        compare_dataframe.ix[i - 3, 2] = float(precent) / float(i)
        compare_dataframe.ix[i - 3, 3] = i
    compare_score_max = float('-inf')
    middle = list(compare_dataframe['i'])
    compare_dataframe = (compare_dataframe - compare_dataframe.min()) / (compare_dataframe.max() - compare_dataframe.min())
    compare_dataframe = compare_dataframe.fillna(0)
    compare_dataframe['i'] = middle
    for i in np.array(compare_dataframe):
        compare_score = i[0] - i[1] - 0.5 * i[2]
        if compare_score_max < compare_score:
            compare_score_max = compare_score
            circle = int(i[3])
    return circle


def calc_bodong(day, period, circle, data_new, p, limit_up=1.5, limit_down=0.9, calc_avg=0):
    bodongzhi = np.zeros((period, circle), dtype=float)
    for i in xrange(period):
        start_time2_index = len(day) - 1 - (i + 1) * circle
        end_time2_index = len(day) - i * circle
        data_bodong = np.array(data_new.ix[(start_time2_index + 1):end_time2_index, p])
        for j in xrange(circle):
            bodongzhi[i, j] = float((data_bodong[j] - data_bodong.mean())) / float(data_bodong.mean())
    bodongzhi_final = np.zeros(circle,)
    weight_bodong = np.zeros(period,)
    for i in xrange(period):
        weight_bodong[i] = math.exp(-0.35 * i)
    if calc_avg == 0:
        avge = 1
    else:
        avge = float(sum(weight_bodong))
    for i in xrange(circle):
        bodongzhi_final[i] = bodongzhi[:, i].dot(weight_bodong.T) / avge
    for i in xrange(len(bodongzhi_final)):
        if bodongzhi_final[i] < -limit_down:
            bodongzhi_final[i] = -limit_down
        if bodongzhi_final[i] > limit_up:
            bodongzhi_final[i] = limit_up
    return bodongzhi_final


def calc_bodong_new(day, period, circle, data_new, p, limit_up=1.5, limit_down=0.9, calc_avg=0):
    bodongzhi = np.zeros((period, circle), dtype=float)
    for i in xrange(period):
        start_time2_index = len(day) - 1 - (i + 1) * circle
        end_time2_index = len(day) - i * circle
        data_bodong = np.array(data_new.ix[(start_time2_index + 1):end_time2_index, p])
        lr = LinearRegression(n_jobs=-1)
        lr.fit(np.arange(len(data_bodong))[:, np.newaxis], data_bodong)
        lr_fit = lr.predict(np.arange(len(data_bodong))[:, np.newaxis])
        for j in xrange(circle):
            bodongzhi[i, j] = float((data_bodong[j] - data_bodong.mean())) / float(lr_fit[j])
    bodongzhi_final = np.zeros(circle,)
    weight_bodong = np.zeros(period,)
    for i in xrange(period):
        weight_bodong[i] = math.exp(-0.35 * i)
    if calc_avg == 0:
        avge = 1
    else:
        avge = float(sum(weight_bodong))
    for i in xrange(circle):
        bodongzhi_final[i] = bodongzhi[:, i].dot(weight_bodong.T) / avge
    for i in xrange(len(bodongzhi_final)):
        if bodongzhi_final[i] < -limit_down:
            bodongzhi_final[i] = -limit_down
        if bodongzhi_final[i] > limit_up:
            bodongzhi_final[i] = limit_up
    return bodongzhi_final


def adf_test(data):
    x = np.zeros(len(data),)
    for i in xrange(len(data)):
        x[i] = np.array(data)[i][0]
    dftest = adfuller(x, autolag='BIC')
    return dftest[1]


def balance_data(data, predict, balance_day):
    output = copy.deepcopy(predict)
    balance_new = sum(predict[:balance_day]) / balance_day
    balance_old = sum(data.tail(balance_day)) / balance_day
    output += balance_old - balance_new
    return output


# 模型开关
balance = 0
balance_day = 3
linear_model_switch = 1
FM_model_switch = 0  # 只有linear_model_switch为1时才有效
GBDT_model_switch = 0
RF_model_switch = 0
XGB_model_switch = 0
Arima_model_switch = 0
Arima_pf = 0  #为0时使用statsmodel的时间序列，1时使用pyflux的
# 模型参数
linear_bodong = 0  # 一旦此项为1时，使用90天线性模型+波动(非乘法)
linear_bodong_guize_FM = 0  # 只有linear_bodong为1时有效，当此项为0时，使用规则计算波动(与所有线性模型都已融合)，当此项为1时，使用FM模型作为波动
auto_ARIMA_diff = 0
arima_p_max = 4
arima_q_max = 4
arima_linear_mix = 1
# 输出图像
show_single_picture = 0
linear_model_figure = 0
Tree_figure_output = 0
arima_figure = 0
finial_figure = 0  # 最终数据处理
total_scale = 1
total_reduce = 0

test = 0

data = pd.read_csv('./origin_data.csv', encoding='utf8', header=0)
data[u'日期'] = pd.to_datetime(data[u'日期'])
region = pd.DataFrame(data[u'地区'].drop_duplicates())
region['region'] = np.arange(len(region))
date = data[u'日期'].drop_duplicates()
date = date.reset_index(drop=True)
predict_date = pd.date_range('20170101', periods=90)
final_example = pd.DataFrame(np.zeros((len(predict_date), 3), dtype=float), columns=[u'日期', 'region', u'价格'])
final_example[u'日期'] = predict_date
data = pd.merge(data, region, on=u'地区', how='left')
del data[u'地区']
final = pd.DataFrame(np.zeros((len(predict_date) * len(region), 3), dtype=float), columns=[u'日期', u'地区', u'价格'])
final[u'地区'] = np.array(len(predict_date) * list(region['region']))
for i in xrange(len(predict_date)):
    final.ix[i * len(region):(i + 1) * len(region), 0] = predict_date[i]

if test == 1:
    region_run = [6]
else:
    region_run = region['region']

GBDT_score_list = []
RF_score_list = []
XGB_score_list = []
arima_skip_list = []
arima_list = []
for k in region_run:
    predict_gbdt = np.zeros(30,)
    predict_RF = np.zeros(30,)
    predict_xgb = np.zeros(30,)
    final_new = copy.deepcopy(final_example)
    final_new['region'] = k
    data_new = data[data['region'] == k]
    data_new = data_new.reset_index(drop=True)
    data_new2 = pd.DataFrame(np.zeros((len(date), 1), dtype=float), columns=[u'日期'])
    data_new2[u'日期'] = date
    data_new = pd.merge(data_new2, data_new, on=u'日期', how='left')
    data_null_date = data_new[u'日期'][data_new[u'价格'].isnull()].index
    for i in data_null_date:
        date2 = np.arange(i - 3, i)
        data_new.ix[i, u'价格'] = float(sum([data_new.ix[j, 1] for j in date2]) / len(date2))
        data_new.ix[i, u'数量'] = float(sum([data_new.ix[j, 2] for j in date2]) / len(date2))
        data_new.ix[i, u'均重'] = float(sum([data_new.ix[j, 3] for j in date2]) / len(date2))

    if show_single_picture == 1:
        plt.figure()
        L1, = plt.plot(data_new[u'日期'], data_new[u'价格'])
        L2, = plt.plot(data_new[u'日期'], data_new[u'数量'])
        L3, = plt.plot(data_new[u'日期'], data_new[u'均重'])
        plt.legend(handles=[L1, L2, L3], labels=['price', 'number', 'weight'], loc='best')
        plt.show()

    # 计算小周期
    circle_small = calc_circle(date, 3, 20, data_new, 1)
    circle_big = calc_circle(date, 50, 200, data_new, 1)

    if linear_model_switch == 1:
        y1 = data_new.ix[:, 2]
        y_1 = np.array(y1)
        y1 = y1[:, np.newaxis]

        y2 = data_new.ix[:, 3]
        y2 = np.array(y2)
        y2 = y2[:, np.newaxis]

        x = np.arange(len(date))
        x = x[:, np.newaxis]
        y = np.array(data_new.ix[:, 1])
        y = y[:, np.newaxis]
        x_all = np.array(data_new.ix[:, 2:4])
        pre = np.arange(len(data_new), len(data_new) + len(predict_date))
        pre = pre[:, np.newaxis]
        lr = LinearRegression(n_jobs=-1)

        lr.fit(x, y)
        y_1 = lr.predict(pre)
        y_1_all = lr.predict(x)

        lr.fit(x, y1)
        y_2 = lr.predict(pre)
        y_2_all = lr.predict(x)

        lr.fit(x, y2)
        y_3 = lr.predict(pre)
        y_3_all = lr.predict(x)

        lr.fit(x[(len(date) - 90):], y[(len(date) - 90):])
        y_4 = lr.predict(pre)
        y_4_all = lr.predict(x)

        # 线性融合
        y_linear_weight = [1, 9]
        for i in xrange(len(predict_date)):
            y_1[i] = (float(y_linear_weight[0] * y_1[i]) + float(y_linear_weight[1] * y_4[i])) / float(sum(y_linear_weight))

        # 计算波动
        period_small = 15
        bodong_small = calc_bodong_new(date, period_small, circle_small, data_new, 1, calc_avg=1)
        period_big = 5
        bodong_big = calc_bodong_new(date, period_big, circle_big, data_new, 1, calc_avg=1)

        predict_linear = np.zeros(len(predict_date),)
        bodong_small_final = np.array(30 * list(bodong_small))[:len(predict_date)]
        bodong_big_final = np.array(2 * list(bodong_big))[:len(predict_date)]
        for i in xrange(len(predict_date)):
            predict_linear[i] = float(y_1[i] * (1 + bodong_small_final[i]) * (1 * (1 + bodong_big_final[i])))
            if predict_linear[i] < 0:
                predict_linear[i] = int(0)
        if balance == 1:
            predict_linear = balance_data(data_new[u'价格'], predict_linear, balance_day)

        period_number_weight = 10
        circle_number = calc_circle(date, 3, 20, data_new, 2)
        bodong_number = calc_bodong_new(date, period_small, circle_number, data_new, 2) / 2
        circle_weight = calc_circle(date, 3, 20, data_new, 3)
        bodong_weight = calc_bodong_new(date, period_small, circle_weight, data_new, 3) / 2
        bodong_number2 = calc_bodong_new(date, period_number_weight, 4 * circle_number, data_new, 2) / 2
        bodong_weight2 = calc_bodong_new(date, period_number_weight, 4 * circle_weight, data_new, 3) / 2

        predict_number = np.zeros(len(predict_date),)
        bodong_number_final = np.array(30 * list(bodong_number))[:len(predict_date)]
        bodong_number2_final = np.array(10 * list(bodong_number2))[:len(predict_date)]
        for i in xrange(len(predict_date)):
            predict_number[i] = float(y_2[i] * (1 + bodong_number_final[i]) * (1 + bodong_number2_final[i]))
            if predict_number[i] < 0:
                predict_number[i] = int(0)

        predict_weight = np.zeros(len(predict_date),)
        bodong_weight_final = np.array(30 * list(bodong_weight))[:len(predict_date)]
        bodong_weight2_final = np.array(10 * list(bodong_weight2))[:len(predict_date)]
        for i in xrange(len(predict_date)):
            predict_weight[i] = float(y_3[i] * (1 + bodong_weight_final[i]) * (1 + bodong_weight2_final[i]))
            if predict_weight[i] < 0:
                predict_weight[i] = int(0)

        x_all_predict = np.concatenate([predict_number[:, np.newaxis], predict_weight[:, np.newaxis]], axis=1)
        x_all = np.concatenate([x_all, x_all_predict], axis=0)
        x_all = np.concatenate([x_all, np.arange(1, len(date) + len(predict_date) + 1)[:, np.newaxis]], axis=1)
        x_all = pd.DataFrame(x_all, columns = ['weight', 'number', 'day'])
        x_all['circle'] = 0
        circle_small_iter = copy.deepcopy(circle_small)
        for i in xrange(len(date)):
            if circle_small_iter == 0:
                circle_small_iter = copy.deepcopy(circle_small)
            x_all.ix[len(date) - 1 - i,'circle'] = circle_small_iter
            circle_small_iter -= 1
        for i in xrange(len(predict_date)):
            if circle_small_iter == 0:
                circle_small_iter = copy.deepcopy(circle_small)
            x_all.ix[len(date) - 1 - i,'circle'] = circle_small_iter
            circle_small_iter -= 1
        x_all['fix_w_n'] = x_all['weight'] * x_all['number']
        x_all['fix_w_c'] = x_all['weight'] * x_all['circle']
        x_all['fix_n_c'] = x_all['number'] * x_all['circle']
        lr.fit(x_all[:len(date), :], y)
        y_all = lr.predict(x_all)
        y_all_predict = y_all[len(date):, :]

        linear_weight = [5, 5]
        y_linear = np.zeros(len(predict_date),)
        for i in xrange(len(predict_date)):
            y_linear[i] = float(predict_linear[i] * linear_weight[0] + y_all_predict[i] * linear_weight[1]) / float(sum(linear_weight))

        if FM_model_switch == 1:
            FM_model = FactorizationMachineRegressor(n_components=4, alpha=0, beta=0, init_lambdas='random_signs', max_iter=1000, verbose=False)
            FM_model.fit(x_all[:len(date), :], y)
            predict_FM = FM_model.predict(x_all[len(date):, :])
        else:
            predict_FM = np.zeros(len(predict_date),)

        if linear_bodong == 1:
            if linear_bodong_guize_FM == 0:
                calc_bodong_data = y_linear
            elif linear_bodong_guize_FM == 1:
                calc_bodong_data = predict_FM
            y_all_bodong = calc_bodong_data - np.mean(calc_bodong_data)
            y_4 += y_all_bodong[:, np.newaxis]
            y_linear = y_4

        if linear_model_figure == 1:
            plt.figure()
            L1, = plt.plot(data_new.ix[:, 0], data_new.ix[:, 1])
            L2, = plt.plot(data_new.ix[:, 0], data_new.ix[:, 2])
            L3, = plt.plot(data_new.ix[:, 0], data_new.ix[:, 3])
            L4, = plt.plot(predict_date, y_1)
            L5, = plt.plot(predict_date, y_2)
            L6, = plt.plot(predict_date, y_3)
            L7, = plt.plot(predict_date, predict_linear)
            L8, = plt.plot(predict_date, predict_number)
            L9, = plt.plot(predict_date, predict_weight)
            L10, = plt.plot(data_new.ix[:, 0], y_all[:len(date), :])
            L11, = plt.plot(predict_date, y_all_predict)
            L12, = plt.plot(predict_date, y_linear)
            L13, = plt.plot(predict_date, y_4)
            L14, = plt.plot(predict_date, predict_FM, linestyle='--')
            plt.legend(handles=[L1, L2, L3, L4, L5, L6, L7, L10, L11, L12, L13, L14], labels=['price', 'number', 'weight', 'predict price', 'predict number', 'predict weight', 'predict linear', 'predict num+weight', 'predict num+weight 90', 'final linear predict', '90day predict', 'FM predict'], loc='best')
            plt.show()

    if GBDT_model_switch == 1 or RF_model_switch == 1 or XGB_model_switch == 1:
        circle_tree = copy.deepcopy(circle_big)
        circle_tree_small = copy.deepcopy(circle_small)
        while circle_tree < 90:
            circle_tree *= 2
        period_tree = int((len(data_new) / circle_tree) - 2)
        weight_tree = np.zeros(circle_tree * period_tree,)
        for i in xrange(period_tree):
            weight_tree[i * circle_tree:(i + 1) * circle_tree] = math.exp(-0.85 * i)
        data_tree = copy.deepcopy(data_new)
        data_tree['circle_small'] = 0
        for i in xrange(len(data_tree)):
            if circle_tree_small == 0:
                circle_tree_small = circle_small
            data_tree.ix[len(data_tree) - 1 - i, 'circle_small'] = circle_tree_small
            circle_tree_small -= 1
        del data_tree[u'日期'], data_tree['region']
        train_tree = pd.DataFrame(np.zeros((0, 0), dtype=float))
        for i in xrange(period_tree):
            train_tree_middle = data_tree.ix[(len(data_new) - (i + 2) * circle_tree):(len(data_new) - (i + 1) * circle_tree - 1), :].reset_index(drop=True)
            train_tree_middle['x'] = np.arange(1, circle_tree + 1)
            train_tree_label = data_tree.ix[(len(data_new) - (i + 1) * circle_tree):(len(data_new) - 1 - i * circle_tree), 0].reset_index(drop=True)
            train_tree_middle = pd.concat([train_tree_middle, train_tree_label], axis=1)
            train_tree = pd.concat([train_tree, train_tree_middle], axis=0)

        test_tree = data_tree.ix[(len(data_new) - circle_tree):, :].reset_index(drop=True)
        test_tree['x'] = np.arange(1, circle_tree + 1)
        train_tree = np.array(train_tree)
        test_tree = np.array(test_tree)
        x_tree = train_tree[:, 1:]
        y_tree = train_tree[:, 0]

        if GBDT_model_switch == 1:
            gbdt = GradientBoostingRegressor(learning_rate=0.01, n_estimators=2000, max_depth=5)
            gbdt.fit(x_tree, y_tree, sample_weight=weight_tree)
            gbdt_feature_score = gbdt.feature_importances_
            GBDT_score_list.append([k, gbdt_feature_score])
            predict_gbdt = gbdt.predict(test_tree)[:len(predict_date)]

        if RF_model_switch == 1:
            RF = RandomForestRegressor(n_estimators=500, max_depth=5, oob_score=True, n_jobs=-1)
            RF.fit(x_tree, y_tree, sample_weight=weight_tree)
            RF_feature_score = RF.feature_importances_
            RF_score_list.append([k, RF_feature_score])
            predict_RF = RF.predict(test_tree)[:len(predict_date)]

        if XGB_model_switch == 1:
            train_xgb = xgb.DMatrix(x_tree, label=y_tree, weight=weight_tree)
            predict_xgb = xgb.DMatrix(test_tree)
            params = {
                'booster': 'gbtree',
                #'objective': 'rank:pairwise',
                'objective': 'reg:linear',
                'eval_metric': 'rmse',
                'gamma': 0.1,
                'slient': 0,
                'min_child_weight': 1,
                'max_depth': 15,
                'lambda': 1,
                'subsample': 0.7,
                'colsample_bytree': 0.7,
                'colsample_bylevel': 0.7,
                'eta': 0.01,
                'tree_method': 'exact',
                'seed': 0,
                'nthread': 16
            }
            watchlist = [(train_xgb, 'train')]
            watchlist2 = {}
            xgb_model = xgb.train(params, train_xgb, num_boost_round=5000, early_stopping_rounds=30, evals=watchlist, evals_result=watchlist2)
            XGB_score_list.append([k, xgb_model.get_fscore(), watchlist2['train']])
            predict_xgb = xgb_model.predict(predict_xgb)[:len(predict_date)]

        if Tree_figure_output == 1:
            plt.figure()
            L1, = plt.plot(data_new.ix[:, 0], data_new.ix[:, 1])
            L2, = plt.plot(predict_date, predict_gbdt)
            L3, = plt.plot(predict_date, predict_RF)
            L4, = plt.plot(predict_date, predict_xgb)
            plt.legend(handles=[L1, L2, L3, L4, ], labels=['true data', 'GBDT predict data', 'RF predict data', 'XGB predict data'])
            plt.show()

    if Arima_model_switch == 1:
        arima_skip = 0
        data_arima_origin = pd.DataFrame(data_new.ix[:, 1])
        data_arima_origin.columns = ['price']
        data_arima_origin_time = data_arima_origin.set_index(date)
        arima_data_dftest_p = adf_test(data_arima_origin)
        data_new_diff_1 = pd.DataFrame(data_new.ix[:, 1].diff(1).dropna())
        data_new_diff_2 = pd.DataFrame(data_new.ix[:, 1].diff(2).dropna())
        if auto_ARIMA_diff == 1:
            if arima_data_dftest_p == None:
                arima_skip = 1
                arima_skip_list.append([k])
            elif arima_data_dftest_p > 0.01:
                '''
                arima_decomposition = seasonal_decompose(data_arima_origin_time,freq=1)
                arima_trend = arima_decomposition.trend
                arima_seasonal = arima_decomposition.seasonal
                arima_residual = arima_decomposition.resid
                arima_data_decompose_dftest_p = adf_test(np.array(arima_residual.dropna()))
                '''
                arima_data_decompose_dftest_p = adf_test(np.array(data_new_diff_1.dropna()))
                arima_data_decompose_dftest_p_2 = adf_test(np.array(data_new_diff_2.dropna()))
                if arima_data_decompose_dftest_p <= 0.01 or arima_data_decompose_dftest_p_2 <= 0.01:
                    if arima_data_decompose_dftest_p <= arima_data_decompose_dftest_p_2:
                        d = 1
                    else:
                        d = 2
                else:
                    arima_skip = 1
            else:
                d = 0
        else:
            d = [0, 1, 2]

        if Arima_pf == 0:
            arima_p = 0
            arima_q = 0
            arima_bic = float('inf')
            arima_model = None
            for diff_num in d:
                for p in xrange(0, arima_p_max + 1):
                    for q in xrange(0, arima_q_max + 1):
                        # print p,q,i
                        arima_model_middle = ARIMA(data_arima_origin, order=(p, diff_num, q), dates=date)
                        try:
                            arima_results_ARIMA = arima_model_middle.fit(disp=-1)
                        except:
                            continue
                        arima_results_bic = arima_results_ARIMA.bic
                        if arima_results_bic < arima_bic:
                            arima_p = p
                            arima_q = q
                            arima_model = arima_results_ARIMA
                            arima_bic = arima_results_bic
                            arima_diff = diff_num
                if arima_model == None:
                    predict_arima = np.zeros(len(predict_date),)
                else:
                    predict_arima = arima_model.forecast(len(predict_date))[0]
                    arima_list.append([k, arima_p, arima_q, arima_diff, 'stat'])
            predict_arima_pf = np.zeros(len(predict_date),)
            
        if Arima_pf == 1:
            arima_p = 0
            arima_q = 0
            arima_bic = float('inf')
            arima_model = None
            for diff_num in d:
                for p in xrange(0, arima_p_max + 1):
                    for q in xrange(0, arima_q_max + 1):
                        # print p,q,i
                        arima_model_middle = pf.ARIMA(data=data_arima_origin, ar=p, integ=diff_num, ma=q)
                        try:
                            arima_results_ARIMAX = arima_model_middle.fit(method='MLE')
                        except:
                            continue
                        arima_results_bic = arima_results_ARIMAX.bic
                        if arima_results_bic < arima_bic:
                            arima_p = p
                            arima_q = q
                            arima_model = arima_model_middle
                            arima_bic = arima_results_bic
                            arima_diff = diff_num
                if arima_model == None:
                    predict_arima_pf = np.zeros(len(predict_date),)
                else:
                    predict_arima_pf = arima_model.predict(h=len(predict_date))
                    arima_list.append([k, arima_p, arima_q, arima_diff, 'pf'])
            predict_arima = np.zeros(len(predict_date),)

        data_arima_origin = np.array(map((lambda x: x[0]), np.array(data_arima_origin)))
        GARCH_model = pf.GARCH(data=np.array(data_arima_origin), p=2, q=2)
        GARCH_model.fit('MLE')
        # model.plot_fit()
        predict_garch = GARCH_model.predict(90)

        predict_arima_origin = copy.deepcopy(predict_arima)
        for i in xrange(len(predict_date)):
            predict_arima[i] = float(predict_arima[i] * (1 + bodong_small_final[i]) * (1 * (1 + bodong_big_final[i])))
            if predict_arima[i] < 0:
                predict_arima[i] = int(0)

        if arima_linear_mix == 1:
            arima_linear_weight = [1, 9]
            for i in xrange(len(predict_date)):
                predict_arima[i] = float(predict_arima[i] * arima_linear_weight[0] + y_all_predict[i] * arima_linear_weight[1]) / float(sum(arima_linear_weight))

    final_new = predict_arima_origin
    #final_new = y_4
    for i in xrange(len(predict_date)):
        final[u'价格'][(final[u'地区'] == k) & (final[u'日期'] == predict_date[i])] = final_new[i]
    if balance == 1:
        final[u'价格'][final[u'地区'] == k] = total_scale * balance_data(data_new[u'价格'], final_new, balance_day) - total_reduce
        final_new = total_scale * balance_data(data_new[u'价格'], final_new, balance_day) - total_reduce

    print('shop %s finished' % (k + 1))

    if arima_figure == 1:
        # model.plot_predict(h=90,past_values=len(date))
        plt.figure()
        L1, = plt.plot(data_new.ix[:, 0], data_new.ix[:, 1])
        L2, = plt.plot(predict_date, final_new)
        L3, = plt.plot(predict_date, predict_arima_origin)
        L4, = plt.plot(predict_date, y_linear)
        L5, = plt.plot(predict_date, predict_arima)
        L6, = plt.plot(predict_date, predict_garch)
        L7, = plt.plot(predict_date, predict_arima_pf)
        plt.legend(handles=[L1, L2, L3, L4, L5, L6], labels=['true data', 'final predict', 'Arima predict(nobodong)', 'final linear predict', 'Arima predict', 'garch predict'])
        plt.show()

    if finial_figure == 1:
        plt.figure()
        L1, = plt.plot(data_new.ix[:, 0], data_new.ix[:, 1])
        L2, = plt.plot(predict_date, final_new)
        plt.legend(handles=[L1, L2], labels=['true data', 'predict'])
        plt.show()

final = final.replace({u'地区': {0: '云南', 1: '四川', 2: '安徽', 3: '山东', 4: '广东', 5: '广西', 6: '江苏', 7: '江西', 8: '浙江', 9: '海南', 10: '湖北', 11: '湖南', 12: '福建', 13: '贵州', 14: '重庆'}})
final.to_csv('./predict.csv', index=None, encoding='utf-8')
final = pd.read_csv('./predict.csv', header=0, encoding='utf-8')
for i in xrange(len(final)):
    final.ix[i, 0] = final.ix[i, 0].replace(' 00:00:00', '')
final.to_csv('./predict.csv', index=None, encoding='utf-8')

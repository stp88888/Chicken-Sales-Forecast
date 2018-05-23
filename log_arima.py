# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 16:20:20 2017

@author: xiejiajia
"""


import pandas as pd
import numpy as np
import warnings # current version of seaborn generates a bunch of warnings that we'll ignore
warnings.filterwarnings("ignore")
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt

train = pd.read_csv('origin_data.csv')#86
train = train.replace({
        '地区' : {
                    '云南':0,
                    '四川':1,
                    '安徽':2,
                    '山东':3,
                    '广东':4,
                    '广西':5,
                    '江苏':6,
                    '江西':7,
                    '浙江':8,
                    '海南':9,
                    '湖北':10,
                    '湖南':11,
                    '福建':12,
                    '贵州':13,
                    '重庆':14
        }
    })
train['date']=train['日期']
train['place']=train['地区']
train['label']=train['价格']
train['num']=train['数量']
train['av_weight']=train['均重']
del train['日期'],train['地区'],train['价格'],train['数量'],train['均重']
train['dateo'] = pd.to_datetime(train['date'])  # 将数据类型转换为日期类型
del train['date']

rng = list(pd.date_range('2014-01-01', periods=1096, freq='D'))
ts = pd.DataFrame(np.random.randn(1096),index=rng).reset_index()
ts.columns = [['dateo','aaa']]
ts = list(ts['dateo'])
df = pd.DataFrame(ts, columns=['dateo']) 

data_nonull = None
test_fea_all = []

t = [[0]]*15
for a in range(0,15):     
    data = train[train.place ==a][['dateo','label']]
    data_all = pd.merge(df,data,on='dateo',how='left') 
    data_all = data_all.fillna(method = 'pad').set_index('dateo')
    

    label = list(data_all['label'])
    if a== 0:
        data_all_1 = data_all[400:]#a=0
    if a== 1:
        data_all_1 = data_all[500:]#a=1
    if a== 2:
        data_all_1 = data_all[300:]#a=2
    if a== 3:
        data_all_1 = data_all[400:]#a=3
    if a== 4:
        data_all_1 = data_all[200:]#a=4
    if a== 5:
        data_all_1 = data_all[200:]#a=5
    if a== 6:
        data_all_1 = data_all[200:]#a=6
    if a== 7:
        data_all_1 = data_all[400:]#a=7
    if a== 8:
        data_all_1 = data_all[200:]#a=8
    if a== 9:
        data_all_1 = data_all[400:]#a=9
    if a== 10:
        data_all_1 = data_all[400:]#a=10
    if a== 11:
        data_all_1 = data_all[200:]#a=11
    if a== 12:
        data_all_1 = data_all[200:]#a=12
    if a== 13:
        data_all_1 = data_all[400:]#a=13
    if a== 14:
        data_all_1 = data_all[600:]#a=14
    ts_log = np.log(data_all_1)
    #ts_log.plot()


    pmax = 5
    qmax = 5
    #bic矩阵
    bic_matrix = []
    for p in range(pmax+1):
        tmp = []
        for q in range(qmax+1):
        #存在部分报错，所以用try来跳过报错。
            try:
                tmp.append(ARIMA(ts_log, (p,1,q)).fit().bic)
            except:
                tmp.append(None)
        bic_matrix.append(tmp)
    #从中可以找出最小值
    bic_matrix = pd.DataFrame(bic_matrix)
    #先用stack展平，然后用idxmin找出最小值位置。
    p,q = bic_matrix.stack().idxmin()
    #print bic_matrix
    print(u'商店：%s，BIC最小的p值和q值为：%s、%s' %(a+1,p,q))
    #建立ARIMA(0, 1, 1)模型
    model = ARIMA(ts_log, (p,1,q)).fit()
    #作为期90天的预测，返回预测结果、标准误差、置信区间。
    aaa = np.exp(model.forecast(90)[0])    
    t[a] = aaa




    
    
#print t
t1 = pd.DataFrame(np.array(t))[0]
for i in range(1,90):
    t1 = pd.concat([t1,pd.DataFrame(np.array(t))[i]], axis = 0)
t1=t1.reset_index(drop = True)

rng = pd.date_range('2017-01-01', '2017-03-31', freq='D')
result=pd.DataFrame()
result['date']=rng
result = pd.concat([result,result,result,result,result,result,result,result,result,result,result,result,result,result,result], axis = 0)
result=result.sort_values('date').reset_index(drop = True)

ts1 = pd.Series(range(0,15)*31)
ts2 = pd.Series(range(0,15)*28)
ts=pd.concat([ts1,ts2,ts1], axis = 0).reset_index(drop = True)
result['place']=ts
result['label']=t1

result = result.replace({
        'place' : {
                    0:'云南',
                    1:'四川',
                    2:'安徽',
                    3:'山东',
                    4:'广东',
                    5:'广西',
                    6:'江苏',
                    7:'江西',
                    8:'浙江',
                    9:'海南',
                    10:'湖北',
                    11:'湖南',
                    12:'福建',
                    13:'贵州',
                    14:'重庆'
        }
    })
result['日期']=result['date']
result['地区']=result['place']
result['价格']=result['label']
del result['date'],result['place'],result['label']
result.to_csv('result-xie.csv', index=None)

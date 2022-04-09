import FinanceDataReader as fdr
import numpy as np

# 모듈 불러오기
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import backend as K
import matplotlib.cm as cm 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import math



# 5) eager execution 기능 끄기
tf.compat.v1.disable_eager_execution()


#코스피200 종가데이터 생성(실제데이터)
df = fdr.DataReader('005930','2020')

df = fdr.DataReader('000660','2021')

#로그 수익률
df['log_rtn'] = np.log(df.Close/df.Close.shift(1))
df = df.dropna()[['log_rtn','Close']]

df  = df[:200]


plt.plot(df["Close"])

df_close = df['Close']
df_log = df['log_rtn']


#정규분포 난수 
r_n = np.random.normal(size = len(df))


#변동률 계산
roc = math.sqrt(((df_log - df_log.mean())**2).sum()/(len(df_log)-1))

#평균 수익률
earning_rate_mean = df_log.mean()*100 -0.5*((df_log.mean()*100)*(df_log.mean()*100))



df_close = pd.DataFrame(df_close)
df_close['stock'] = 0

#df_close = df_close[:-1]

import itertools
color_cycle= itertools.cycle(["orange","pink","brown","red","grey","yellow","green"])


for i in range(7):
    r_n = np.random.normal(size = len(df))
    stock_random = np.array([])
    for j in range(len(df_close)):
        stock_random = np.append(stock_random, r_n[j]*(roc*100) + (earning_rate_mean) )
    df_close['stock'] =  stock_random/100
    
    cumsum_list = []
    for k in range(len(df_close)):
        if k == 0:
            a = df_close['Close'][0] * np.exp(df_close['stock'][k])
            cumsum_list.append(a)
        else : 
            b = cumsum_list[-1] * np.exp(df_close['stock'][k])
            cumsum_list.append(b)
            
            
    df_close["cumsum"] = cumsum_list

    plt.plot(df_close[:-1]["cumsum"][:200], color = next(color_cycle), label ="gan data_ %d" %i)
plt.plot(df_close[:200].Close, '-b', label="real data")
plt.legend()

                                                


distribution_list = []
distribution_list_100 = []
for i in range(1000):
    r_n = np.random.normal(size = len(df))
    stock_random = np.array([])
    for j in range(len(df_close)):
        stock_random = np.append(stock_random, r_n[j]*(roc*100) + (earning_rate_mean) )
    df_close['stock'] =  stock_random/100
    
    cumsum_list = []
    for k in range(len(df_close)):
        if k == 0:
            a = df_close['Close'][0] * np.exp(df_close['stock'][k])
            cumsum_list.append(a)
        else : 
            b = cumsum_list[-1] * np.exp(df_close['stock'][k])
            cumsum_list.append(b)
            if k == 100:
                distribution_list_100.append(b)
            
    distribution_list.append(b)


df_close["Close"][100]

len(distribution_list_100)
len(distribution_list)


plt.plot(distribution_list, label='Discriminated Fake Data', color='red')
sns.kdeplot(distribution_list, color='blue', bw=0.3, label='REAL data')
sns.kdeplot(distribution_list_100, color='blue', bw=0.3, label='REAL data')

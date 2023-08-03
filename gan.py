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
from scipy.stats import ks_2samp
from scipy.stats import entropy

# 5) eager execution 기능 끄기
tf.compat.v1.disable_eager_execution()


#코스피200 종가데이터 생성(실제데이터)
df = fdr.DataReader('KS200','2020')

df['log_rtn'] = np.log(df.Close/df.Close.shift(1))
df = df.dropna()[['log_rtn','Close']]

df  = df[58:]


df_close = df['Close']
df_log = df['log_rtn']
df_log = df_log

#변동률 계산
roc = math.sqrt(((df_log - df_log.mean())**2).sum()/(len(df_log)-1))


real_data = df_log.to_numpy()
real_data = real_data.reshape(len(real_data),1) 


"""
# 실제 데이터 준비
real_data = np.random.normal(size=1000)
real_data = real_data.reshape(real_data.shape[0], 1)
"""


# 가짜 데이터 생성
def makeZ(m, n):
    z = np.random.uniform(-1.0, 1.0, size=[m, n])
    return z

# 모델 파라미터 설정
d_input = real_data.shape[1]
d_hidden = 32
d_output = 1 # 주의
g_input = 16
g_hidden = 32
g_output = d_input # 주의

# 옵티마이저 설정
def myOptimizer(lr):
    return RMSprop(learning_rate=lr)

# 1) Discriminator 모델
def build_D():
    d_x = Input(batch_shape=(None, d_input))
    d_h = Dense(d_hidden, activation='relu')(d_x)
    d_o = Dense(d_output, activation='sigmoid')(d_h)
    
    d_model = Model(d_x, d_o)
    d_model.compile(loss='binary_crossentropy', optimizer=myOptimizer(0.001))
    
    return d_model

# 2) Generator 모델
def build_G():
    g_x = Input(batch_shape=(None, g_input))
    g_h = Dense(g_hidden, activation='relu')(g_x)
    g_o = Dense(g_output, activation='linear')(g_h) # 주의
    
    g_model = Model(g_x, g_o) # 주의
    
    return g_model

# 3) GAN 네트워크
def build_GAN(discriminator, generator):
    discriminator.trainable = False # discriminator 업데이트 해제
    z = Input(batch_shape=(None, g_input))
    Gz = generator(z)
    DGz = discriminator(Gz)
    
    gan_model = Model(z, DGz)
    gan_model.compile(loss='binary_crossentropy', optimizer=myOptimizer(0.0005))
    
    return gan_model

# 4) 학습
K.clear_session() # 5) 그래프 초기화

D = build_D() # discriminator 모델 빌드
G = build_G() # generator 모델 빌드
GAN = build_GAN(D, G) # GAN 네트워크 빌드

n_batch_cnt = int(input('입력 데이터 배치 블록 수 설정: '))
n_batch_size = int(real_data.shape[0] / n_batch_cnt)

EPOCHS = int(input('학습 횟수 설정: '))

  
for epoch in range(EPOCHS):
    # 미니배치 업데이트
    for n in range(n_batch_cnt):
        from_, to_ = n*n_batch_size, (n+1)*n_batch_size
        if n == n_batch_cnt -1 : # 마지막 루프
            to_ = real_data.shape[0]
        
        # 학습 데이터 미니배치 준비
        X_batch = real_data[from_: to_]
        Z_batch = makeZ(m=X_batch.shape[0], n=g_input)
        Gz = G.predict(Z_batch) # 가짜 데이터로부터 분포 생성
        
        # discriminator 학습 데이터 준비
        d_target = np.zeros(X_batch.shape[0]*2)
        d_target[:X_batch.shape[0]] = 0.9   
        d_target[X_batch.shape[0]:] = 0.1
        bX_Gz = np.concatenate([X_batch, Gz]) # 묶어줌.
        
        # generator 학습 데이터 준비
        g_target = np.zeros(Z_batch.shape[0])
        g_target[:] = 0.9 # 모두 할당해야 바뀜.
        
        # discriminator 학습        
        loss_D = D.train_on_batch(bX_Gz, d_target) # loss 계산
        
        # generator 학습        
        loss_G = GAN.train_on_batch(Z_batch, g_target)
        
    if epoch % 10 == 0:
        z = makeZ(m=real_data.shape[0], n=g_input)
        fake_data = G.predict(z) # 가짜 데이터 생성
        print("Epoch: %d, D-loss = %.4f, G-loss = %.4f" %(epoch, loss_D, loss_G))
        
    if epoch % 300 == 0 :
        z = makeZ(m=real_data.shape[0], n=g_input)
        fake_data = G.predict(z)
    
        plt.figure(figsize=(8, 5))
        sns.set_style('whitegrid')
        sns.kdeplot(real_data[:, 0], color='blue', bw=0.3, label='REAL data')
        sns.kdeplot(fake_data[:, 0], color='red', bw=0.3, label='FAKE data')
        plt.legend()
        plt.title('REAL vs. FAKE distribution')
        plt.show()



arr1_flat = fake_data.flatten()


kld = entropy(real_data.flatten(), fake_data.flatten())

statistic, p_value = ks_2samp((fake_data*100).flatten(), (real_data*100).flatten())
x = np.linspace(-5, 5, 1000)
cdf_generated = np.cumsum(np.histogram(fake_data, bins=1000, range=(-5, 5), density=True)[0])
cdf_real = np.cumsum(np.histogram(real_data, bins=1000, range=(-5, 5), density=True)[0])

plt.plot(x, cdf_generated, label='Generated Data')
plt.plot(x, cdf_real, label='Real Data')
plt.legend()
plt.title("KS Test: p-value={:.2f}".format(p_value))
plt.show()


# 학습 완료 후 데이터 분포 시각화
z = makeZ(m=real_data.shape[0], n=g_input)
fake_data = G.predict(z)

plt.figure(figsize=(8, 5))
sns.set_style('whitegrid')
sns.kdeplot(real_data[:, 0], color='blue', bw=0.3, label='REAL data')
sns.kdeplot(fake_data[:, 0], color='red', bw=0.3, label='FAKE data')
plt.legend()
plt.title('REAL vs. FAKE distribution')
plt.show()

# 학습 완료 후 discriminator 판별 시각화
d_real_values = D.predict(real_data) # 실제 데이터 판별값
d_fake_values = D.predict(fake_data) # 가짜 데이터 판별값

plt.figure(figsize=(8, 5))
plt.plot(d_real_values, label='Discriminated Real Data')
plt.plot(d_fake_values, label='Discriminated Fake Data', color='red')
plt.title("Discriminator vs. Generator")
plt.legend()
plt.show()



fake_data_list = np.concatenate(fake_data).tolist()

import random


#로그 수익률 생성 함수(input 값: dataframe)
def log_rtn(df):
    df['log_rtn'] = np.log(df.Close/df.Close.shift(1))
    df  = df.dropna()[['log_rtn','Close']]

    
    df['log_rtn'] = df['log_rtn']
    train_log = df['log_rtn']
    return train_log


#정규분포 난수 생성 함수
def random_normal():
    r_n = random.sample(fake_data_list , 1)
    return r_n[0]



#변동률, 평균수익률, 종가 데이터 생성 (n : 며칠동안인지)
def new_data(train):
    train_log = log_rtn(train)
    
    r_n = []
     
    for i in range(len(train)):
        sample = random_normal()
        r_n.append(sample)
    
    
    #일일 수익률(평균)
    rtn_d = train_log.mean()
    
    #변동성(표준편차)
    roc = np.std(train_log)
    
    #평균수익률
    earning_rate_mean = rtn_d -0.5*((roc)**2)
    
    #수익률 
    rtn = earning_rate_mean + roc*np.array(r_n)
    
    
    #새로 생성한 종가 데이터

        
    data = (1000*np.cumprod(np.exp(rtn)))
    
    return data  


data_list = []
a = 100
for i in range(len(rtn)):
    a = a*(np.exp(rtn)[i])
    data_list.append(a)

a = np.exp(rtn)

data_list[1]
99.98512109397925

b = 100*a[0]
b*a[1]

data[:3]

1000 * a


train_log = log_rtn(df)
r_n = []
     
for i in range(len(df)):
        sample = random_normal()
        r_n.append(sample)
    
    
    #일일 수익률(평균)
rtn_d = train_log.mean()
    
    #변동성(표준편차)
roc = np.std(train_log)
    
    #평균수익률
earning_rate_mean = rtn_d -0.5*((roc)**2)
    
    #수익률 
rtn = earning_rate_mean + roc*np.array(r_n)
    
    
    #새로 생성한 종가 데이터
data = (100*np.cumprod(np.exp(rtn)))


sns.kdeplot(fake_data_list, color='blue', bw=0.3, label='REAL data')
sns.kdeplot(r_n, color='blue', bw=0.3, label='REAL data')
sns.kdeplot(rtn , color='blue', bw=0.3, label='REAL data')

rtn.mean()
sum(fake_data_list)/len(fake_data_list)

sum(r_n)/len(r_n)

real_data.mean()



for i in range(100):
    data = new_data(df)
    plt.plot(data[:250])
    plt.legend()

a = []

for i in range(500):
    data = new_data(df)
    a.append(data[:250][-1])

#주가 분포
sns.kdeplot(a , color='blue', bw=0.3, label='200')
plt.legend()



#정규분포 난수 -> GAN 추출 데이터로 대체

fake_data_list = fake_data.reshape(len(fake_data),)

import random
plt.plot(fake_data)

random.choice(fake_data_list )



df_close = pd.DataFrame(df_close)
df_close['stock'] = 0


#수익률 계산
stock_yield = roc * random.choice(fake_data_list ) + df_log.mean()
for i in range(len(df_close)):
    df_close['stock'][i] = roc * random.choice(fake_data_list ) + df_log.mean()

cumsum_list = []
for i in range(len(df_close)):
    a = df_close['Close'][0] * np.exp(df_close['stock'][i])
    cumsum_list.append(a)


len(cumsum_list)
len(df_close)




df_close['stock'] = df_close.Close.shift(1) * np.exp(stock_yield)

df_close = df_close.dropna()


df_close['cumsum'] = cumsum_list[1:]



plt.plot(df_close['cumsum'][-100:], '-r', label='GAN data')
plt.plot(df_close[-100:].Close, '-b', label="real data")
plt.legend()

colors = cm.rainbow(np.linspace(0, 1, 10))




import itertools
color_cycle= itertools.cycle(["orange","pink","brown","red","grey","yellow","green"])
earning_rate_mean = df_log.mean()*100 -0.5*((df_log.mean()*100)*(df_log.mean()*100))

for i in range(7):
    
    stock_random = np.array([])
    for j in range(len(df_close)):
        stock_random = np.append(stock_random, ((roc*100 * (random.choice(fake_data_list) * 100) +(earning_rate_mean) )))
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

    plt.plot(df_close[:-1]["cumsum"][:100], color = next(color_cycle), label ="gan data_ %d" %i)
plt.plot(df_close[:100].Close, '-b', label="real data")
plt.legend()



earning_rate_mean = df_log.mean()*100 -0.5*((df_log.mean()*100)*(df_log.mean()*100))


plt.plot(df_close[:-1].cumsum)

sns.kdeplot(df_close['stock'][ :], color='blue', bw=0.3, label='REAL data')
sns.kdeplot(stock_random[ :], color='blue', bw=0.3, label='REAL data')
stock_random.max()

sns.kdeplot(real_data[:, 0], color='blue', bw=0.3, label='REAL data')
sns.kdeplot(fake_data[:, 0], color='red', bw=0.3, label='fake data')
plt.legend()



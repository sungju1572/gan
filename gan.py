"""
import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
"""

import FinanceDataReader as fdr
import numpy as np

# 모듈 불러오기
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

# 5) eager execution 기능 끄기
tf.compat.v1.disable_eager_execution()

# 실제 데이터 준비
real_data = np.random.normal(size=1000)
real_data = real_data.reshape(real_data.shape[0], 1)

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
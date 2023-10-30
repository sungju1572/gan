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
import FinanceDataReader as fdr
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import talib
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import xgboost as xgb
from xgboost import XGBClassifier
import datetime
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import random
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
warnings.filterwarnings('ignore')



def make_real_data(train_real):
    
    train_real['log_rtn'] = np.log(train_real.Close/train_real.Close.shift(1))
    train_real = train_real.dropna()[['log_rtn','Close']]

    df_close = train_real['Close']
    df_log = train_real['log_rtn']

    #변동률 계산
    roc = math.sqrt(((df_log - df_log.mean())**2).sum()/(len(df_log)-1))

    #실제데이터
    real_data = df_log.to_numpy()
    real_data = real_data.reshape(len(real_data),1) 

    return real_data


#정규분포 난수 -> GAN 추출 데이터로 대체
def make_gan_nan(fake_data):
    fake_data_list = fake_data.reshape(len(fake_data),)
    return fake_data_list





#로그 수익률 생성 함수(input 값: dataframe)
def log_rtn(train):
    train['log_rtn'] = np.log(train.Close/train.Close.shift(1))
    train = train.dropna()[['log_rtn','Close']]

    
    train['log_rtn'] = train['log_rtn']
    train_log = train['log_rtn']
    return train_log




#변동률, 평균수익률, 종가 데이터 생성 (n : 며칠동안인지)
def new_data(train_real):
    train_log = log_rtn(train_real)
    r_n =  np.random.normal(size= len(train_real))
    
    #일일 수익률(평균)
    rtn_d = train_log.mean()
    
    #변동성(표준편차)
    roc = np.std(train_log)
    
    #평균수익률
    earning_rate_mean = rtn_d -0.5*((roc)**2)
    
    #수익률 
    rtn = earning_rate_mean + roc*r_n    
    
    #새로 생성한 종가 데이터
    data = (100*np.cumprod(np.exp(rtn)))
    
    
    return data  

"""
#plot 그려보기
for i in range(10):
    data = new_data(train_real, fake_data_list)
    plt.plot(data[:250])
    plt.legend()
"""


#라벨링 리스트 생성 함수(input : (data, 만들 행 갯수))
def label_list(data, num):
    data_df = pd.DataFrame(data, columns = ["data"])

    ###라벨링
    label_shift = data_df['data'].diff().shift(-1)[:num]
    label_shift_list = []

    for i in range(len(label_shift)):
        if label_shift[i] > 0 :
            label_shift_list.append(1)
        else:
            label_shift_list.append(0)
    

    return label_shift_list




#기술지표 생성함수(input : (data,  만들 행 갯수,time(25)))
def tal(data, num, time):
    label = label_list(data, num)
        
    #기술지표들
    train_apo = talib.APO(data[:num])
    train_cmo = talib.CMO(data[:num])
    train_macd , train_macdsignal , train_macdhist = talib.MACD(data[:num])
    train_mom = talib.MOM(data[:num])
    train_ppo = talib.PPO(data[:num])
    train_roc = talib.ROC(data[:num])
    train_rocp = talib.ROCP(data[:num])
    train_rocr = talib.ROCR(data[:num])
    train_rocr100 = talib.ROCR100(data[:num])
    train_rsi = talib.RSI(data[:num])
    train_fasrk, train_fasrd = talib.STOCHRSI(data[:num])
    train_trix = talib.TRIX(data[:num])

    data = {'APO' : train_cmo[time:],
            'CMO' : train_cmo[time:],
            'MACD' : train_macd[time:],
            'MACDSIGNAL' : train_macdsignal[time:],
            'MACDHIST' : train_macdhist[time:],
            'MOM' : train_mom[time:],
            'PPO' : train_ppo[time:],
            'ROC' : train_roc[time:],
            'ROCP' : train_rocp[time:],
            'ROCR' : train_rocr[time:],
            'ROCR100' : train_rocr100[time:],
            'RSI' : train_rsi[time:],
            'FASRK' : train_fasrk[time:],
            'FASRD' : train_fasrd[time:],
            'TRIX' : train_trix[time:],
            'label' : label[time:]}
    
    #train_data 생성(종가 제외)
    train_data = pd.DataFrame(data)
    train_data = train_data.reset_index(drop=True)
    
    return train_data






#test 데이터 생성
def make_test(test):
    test_df = test["Close"].diff().shift(-1)
    test_df = test_df.dropna()
    
    test_df = test_df[99:]
    
    
    test_df_list = []
    
    for i in range(len(test_df)):
        if test_df[i] > 0 :
            test_df_list.append(1)
        else:
            test_df_list.append(0)
    
    
    len(test_df_list)
    
    
    time = 99
    
    test_apo = talib.APO(test['Close'])
    test_cmo = talib.CMO(test['Close'])
    test_macd , test_macdsignal , test_macdhist = talib.MACD(test['Close'])
    test_mom = talib.MOM(test['Close'])
    test_ppo = talib.PPO(test['Close'])
    test_roc = talib.ROC(test['Close'])
    test_rocp = talib.ROCP(test['Close'])
    test_rocr = talib.ROCR(test['Close'])
    test_rocr100 = talib.ROCR100(test['Close'])
    test_rsi = talib.RSI(test['Close'])
    test_fasrk, test_fasrd = talib.STOCHRSI(test["Close"])
    test_trix = talib.TRIX(test['Close'])
    
    
    len(test_df)
    len(test_trix[99:-1])
    
    
    data = {'APO' : test_cmo[time:-1],
            'CMO' : test_cmo[time:-1],
            'MACD' : test_macd[time:-1],
            'MACDSIGNAL' : test_macdsignal[time:-1],
            'MACDHIST' : test_macdhist[time:-1],
            'MOM' : test_mom[time:-1],
            'PPO' : test_ppo[time:-1],
            'ROC' : test_roc[time:-1],
            'ROCP' : test_rocp[time:-1],
            'ROCR' : test_rocr[time:-1],
            'ROCR100' : test_rocr100[time:-1],
            'RSI' : test_rsi[time:-1],
            'FASRK' : test_fasrk[time:-1],
            'FASRD' : test_fasrd[time:-1],
            'TRIX' : test_trix[time:-1],
            'label' : test_df_list
            }
    
    #test_data 생성(종가 제외)
    test_data = pd.DataFrame(data)
    #test_data = test_data.reset_index(drop=True)

    return test_data


#로지스틱모델
def logistic(X_train, y_train, X_test, y_test):
    np.random.seed(40)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    #print(model.score(X_train, y_train))

    y_pred = model.predict(X_test)
    
    return y_pred
    
#결정트리모델
def DT(X_train, y_train, X_test, y_test):
    np.random.seed(40)
    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    #print(accuracy_score(y_pred, y_test)) #0.4552
    
    return y_pred

#knn 모델
def KNN(X_train, y_train, X_test, y_test):
    np.random.seed(40)
    
    classifier = KNeighborsClassifier(n_neighbors = 5)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    #print(accuracy_score(y_pred, y_test)) #0.5203252032520326

    return y_pred
    
#랜덤포레스트 모델
def RF(X_train, y_train, X_test, y_test):    
    rfc = RandomForestClassifier(random_state=40)
    rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)

    #print(accuracy_score(y_pred, y_test)) #0.532520325203252

    return y_pred

#xgboost 모델
def Xgboost(X_train, y_train, X_test, y_test):
    xgb1 = XGBClassifier(tree_method = "hist", device = "cuda")
    parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
                  'objective':['binary:logistic'],
                  'learning_rate': [.03, 0.05, .07], #so called `eta` value
                  'max_depth': [3, 4, 5],
                  'min_child_weight': [4],
                  'silent': [1],
                  'subsample': [0.7],
                  'colsample_bytree': [0.7],
                  'n_estimators': [500],
                  'silent': [1],
                  "random_state" : [40]}

    xgb_grid = GridSearchCV(xgb1,
                            parameters,
                            cv = 2,
                            n_jobs = 5, 
                            verbose=True)

    xgb_grid.fit(X_train,
             y_train)


    #print(xgb_grid.best_score_)
    print(xgb_grid.best_params_)



    #prediction
    y_pred = xgb_grid.predict(X_test)
    best_params = xgb_grid.best_params_
    acc = accuracy_score(y_pred, y_test) #0.540650406504065

    return y_pred , best_params, acc


#MLP
def MLP(X_train, y_train, X_test, y_test):
    # 신경망 모델 정의
    class NeuralNetwork(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(NeuralNetwork, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.1)
            self.fc2 = nn.Linear(hidden_size, output_size)
            self.sigmoid = nn.Sigmoid()
        
        def forward(self, x):
            out = self.fc1(x)
            out = self.relu(out)
            out = self.dropout(out)
            out = self.fc2(out)
            out = self.sigmoid(out)
            return out

    # 모델 초기화 및 하이퍼파라미터 설정
    input_size = 15  # 입력 피처의 차원
    hidden_size = 128  # 은닉층의 뉴런 수
    output_size = 1  # 출력의 차원 (이진 분류이므로 1)

    model = NeuralNetwork(input_size, hidden_size, output_size)
    criterion = nn.BCELoss()  # 이진 교차 엔트로피 손실
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    X_train_1 = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_1 = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    X_test_1 = torch.tensor(X_test.values, dtype=torch.float32)



    # 모델 훈련
    num_epochs = 2000
    for epoch in range(num_epochs):
        outputs = model(X_train_1 )
        loss = criterion(outputs, y_train_1)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

    # 모델 테스트
    with torch.no_grad():

        test_output = model(X_test_1)
        predictions = (test_output > 0.5).int().numpy()
        print("Predictions:", predictions.flatten())
        
    return predictions.flatten()

#DNN
def DNN(X_train, y_train, X_test, y_test):
    # 신경망 모델 정의
    class NeuralNetwork(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(NeuralNetwork, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.1)
            self.fc2 = nn.Linear(hidden_size, hidden_size_2)
            self.relu = nn.ReLU()
            self.dropout2 = nn.Dropout(0.1)
            self.fc3 = nn.Linear(hidden_size_2, output_size)
            self.sigmoid = nn.Sigmoid()
        
        def forward(self, x):
            out = self.fc1(x)
            out = self.relu(out)
            out = self.dropout(out)
            out = self.fc2(out)
            out = self.relu(out)
            out = self.dropout2(out)
            out = self.fc3(out)
            out = self.sigmoid(out)
            return out

    # 모델 초기화 및 하이퍼파라미터 설정
    input_size = 15  # 입력 피처의 차원
    hidden_size = 128  # 은닉층의 뉴런 수
    hidden_size_2 = 128  # 은닉층의 뉴런 수
    output_size = 1  # 출력의 차원 (이진 분류이므로 1)

    model = NeuralNetwork(input_size, hidden_size, output_size)
    criterion = nn.BCELoss()  # 이진 교차 엔트로피 손실
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    X_train_1 = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_1 = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    X_test_1 = torch.tensor(X_test.values, dtype=torch.float32)



    # 모델 훈련
    num_epochs = 2000
    for epoch in range(num_epochs):
        outputs = model(X_train_1 )
        loss = criterion(outputs, y_train_1)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

    # 모델 테스트
    with torch.no_grad():

        test_output = model(X_test_1)
        predictions = (test_output > 0.5).int().numpy()
        print("Predictions:", predictions.flatten())
        
    return predictions.flatten()






#예측값 지표들 추출 (test_data_drop 부분 함수 아직 미완성)
def pred(test_close, y_pred):
    #pred
    len(y_pred)

    test_data_pred = pd.DataFrame(test_close,columns=['Close'])
    test_data_pred = test_data_pred.reset_index(drop=True)

    
    test_data_rtn = test_data_pred['Close'].pct_change() * 100


    test_data_pred["rtn"] = test_data_rtn  

    test_data_pred['label'] = y_pred

    test_data_pred = test_data_pred[1:].dropna().reset_index()



    

    test_data_pred['position'] = None


    for i in range(0, len(test_data_pred)):
            try:
                if test_data_pred['label'][i]+test_data_pred['label'][i+1]==0:
                    test_data_pred['position'][i+1]='no action'
                elif test_data_pred['label'][i]+test_data_pred['label'][i+1]==2:
                    test_data_pred['position'][i+1]='holding'
                elif test_data_pred['label'][i] > test_data_pred['label'][i+1]:
                    test_data_pred['position'][i+1]='sell'
                else:
                    test_data_pred['position'][i+1]='buy'
            except:
                pass


    test_data_pred['position'].iloc[-1] ="sell"

    if test_data_pred['position'][0] == None:
        if test_data_pred['label'][0] ==   1:
            test_data_pred['position'][0] = "buy"




    test_data_pred["new_rtn"] = 0.0

    for i in range(len(test_data_pred)):
        if test_data_pred["position"][i] == "buy" or test_data_pred["position"][i] == "no action" :
            test_data_pred["new_rtn"][i] = 0
        else : 
            test_data_pred["new_rtn"][i] = test_data_pred['rtn'][i] 
             
            
    test_data_pred["new_rtn"].sum()


    #diff
    test_data_pred["diff"] = test_data_pred["Close"].diff()


    #거래 횟수
    test_data_pred['new_diff'] = 0.0

    for i in range(len(test_data_pred)):
        if test_data_pred["position"][i] == "buy" or test_data_pred["position"][i] == "no action" :
            test_data_pred["new_diff"][i] = 0
        else : 
            test_data_pred["new_diff"][i] = test_data_pred['diff'][i] 
             



    #sell 기준 합치기

    test_data_pred["diff_sum"] = 0.0

    a = []

    for i in range(1, len(test_data_pred)):
        if test_data_pred["position"][i] == "holding" :
                a.append(test_data_pred["new_diff"][i])
        elif test_data_pred["position"][i] == "sell"  :
            a.append(test_data_pred["new_diff"][i])
            test_data_pred["diff_sum"][i] = sum(a)
            a=[]


    #rmfovm


    test_data_pred["index"] = test_data_pred.index

    buy_index =[]
    sell_index = []




    for i in range(len(test_data_pred)):
        if test_data_pred["position"][i] == "buy":
            buy_index.append(test_data_pred['index'][i])
        elif test_data_pred['position'][i] == "sell" :       
            sell_index.append(test_data_pred['index'][i])

    len(sell_index)



    #win rate

    win_rate_count = 0

    for i in range(len(test_data_pred)): 
        if test_data_pred["diff_sum"][i] > 0:
            win_rate_count +=1        
        

    #win_rate_count / len(sell_index)



    #payoff_rate
    gain_list = []
    loss_list = []
    all_list = []

    for i in range(len(test_data_pred)):
        if test_data_pred["diff_sum"][i] > 0:
            gain_list.append(test_data_pred["diff_sum"][i])
            all_list.append(test_data_pred["diff_sum"][i])
        elif test_data_pred["diff_sum"][i] <0:
            loss_list.append(test_data_pred["diff_sum"][i])
            all_list.append(test_data_pred["diff_sum"][i])
            
    np.mean(loss_list) / np.mean(gain_list)


    #profit factor

    #sum(loss_list) / sum(gain_list)

    #average gain & loss
    np.mean(gain_list)
    np.mean(loss_list)

    #총손실
    sum(loss_list)

    #총수익
    sum(gain_list)


    #수익률 붙이기


    test_data_rtn = test_close.pct_change() * 100
    len(test_data_rtn)

    test_close_rtn = test_data_rtn[2:].reset_index()["Close"]


    test_data_pred["rtn"] = test_close_rtn 





    test_data_pred["new_rtn"] = 0.0




    for i in range(len(test_data_pred)):
        if test_data_pred["position"][i] == "buy" or test_data_pred["position"][i] == "no action" :
            test_data_pred["new_rtn"][i] = 0
        else : 
            test_data_pred["new_rtn"][i] = test_data_pred['rtn'][i] 
             
            
    test_data_pred["new_rtn"].sum()


    #지표들
    #print("거래횟수 : ", len(sell_index))
    #print("winning ratio :", win_rate_count / len(sell_index))
    #print("평균 수익 :", np.mean(gain_list))
    #print("평균 손실 :", np.mean(loss_list))
    #print("payoff ratio :", np.mean(loss_list) / np.mean(gain_list))
    #print("총수익:", sum(gain_list))
    #print("총손실:", sum(loss_list))
    #print("profit factor:", sum(loss_list) / sum(gain_list))
    
    trade_count = len(sell_index)
    if len(sell_index) != 0:
        winning_ratio = win_rate_count / len(sell_index)
    else:
        winning_ratio = 0
    mean_gain = np.mean(gain_list)
    mean_loss = np.mean(loss_list)
    if np.mean(loss_list) != 0 :
        payoff_ratio = np.mean(gain_list) /  np.mean(loss_list) 
    else:
        payoff_ratio = 0
    sum_gain = sum(gain_list)
    sum_loss = sum(loss_list)
    if sum(loss_list) != 0:
        profit_factor = sum(gain_list) / sum(loss_list) 
    else :
        profit_factor  = 0
    
    return trade_count, winning_ratio, mean_gain , abs(mean_loss), abs(payoff_ratio), sum_gain , abs(sum_loss) , abs(profit_factor)



import warnings
warnings.filterwarnings('ignore')

#데이터 자동화
#epochs_list = [900,1200,1500]
epochs_list = [1500]
kospi_200_list = ["005930","000660","051910","006400","035420","005380","035720","000270","068270","028260","005490","105560","012330","096770","055550","034730","066570","015760","034020","003550","003670","032830","011200","086790","017670","051900","010130","033780","010950","003490"]
#kospi_200_list = ["028260","005490","105560"]
#kospi_200_list = ["012330","096770","055550","034730","066570","015760","034020","003550","003670","032830","011200","086790","017670","051900","010130","033780","010950","003490"]

count_list = [20,50,100,150,200]

train_count = [3,4,5]


#result_df = pd.DataFrame(columns=["ticker","epoch", "count","model", "trade_count", "winning_ratio", "mean_gain", "mean_loss", "payoff_ratio" , "sum_gain" , "sum_loss" , "profit_factor"])
xgboost_parems = {}

for count in train_count :
    result_df = pd.DataFrame(columns=["ticker","epoch", "count","model", "trade_count", "winning_ratio", "mean_gain", "mean_loss", "payoff_ratio" , "sum_gain" , "sum_loss" , "profit_factor"])
    
    for i in kospi_200_list:
        for e in epochs_list:
            if count == 3 :
                train = fdr.DataReader(symbol= i, start='2017', end='2020')
                test = fdr.DataReader(symbol= i ,start='2019', end='2021')

                test_data = make_test(test)
        

                test_data = test_data.reset_index()
            

                test_data = test_data[147:]
            
                test_data = test_data.drop(['Date'], axis=1)
            
        
                test_close = test["Close"]
                test_close  = test_close[246:-1]
                
                train_real =  train

            elif count == 4 :
                train = fdr.DataReader(symbol= i, start='2018', end='2021')
                test = fdr.DataReader(symbol= i ,start='2020', end='2022')

                test_data = make_test(test)
        

                test_data = test_data.reset_index()
            

                test_data = test_data[149:]
            
                test_data = test_data.drop(['Date'], axis=1)
            
        
                test_close = test["Close"]
                test_close  = test_close[248:-1]
                
                train_real =  train
                
            elif count == 5 :
                train = fdr.DataReader(symbol= i, start='2019', end='2022')
                test = fdr.DataReader(symbol= i ,start='2021', end='2023')

                test_data = make_test(test)
        

                test_data = test_data.reset_index()
            

                test_data = test_data[149:]
            
                test_data = test_data.drop(['Date'], axis=1)
            
        
                test_close = test["Close"]
                test_close  = test_close[248:-1]
                
                train_real = train
     
            

            
    
            for c in count_list:
                
                train_data = pd.DataFrame()
                    
                for k in range(c):    
                    data = new_data(train_real )
                    df = tal(data, 338, 88)
                    train_data = pd.concat([train_data, df])
                    
                train_data = train_data.reset_index(drop=True)   
                    
                
                
                #train /test 라벨 나누기
                X_train = train_data.drop(["label"], axis = 1 ) #학습데이터
                y_train = train_data["label"] #정답라벨
                X_test = test_data.drop(['label'], axis=1) #test데이터
                y_test = test_data["label"]
                
                #로지스틱
                y_pred_lg = logistic(X_train, y_train, X_test, y_test)

                    
                #DT
                y_pred_dt = DT(X_train, y_train, X_test, y_test)
        
                #rf
                y_pred_rf = RF(X_train, y_train, X_test, y_test)
                    
                    
                #xgboost
                y_pred_xg, best_params, acc = Xgboost(X_train, y_train, X_test, y_test)
                    
                xgboost_parems[i + "_" + str(c)] =  best_params
                
                #MLP
                y_pred_mlp = MLP(X_train, y_train, X_test, y_test)
                
                #DNN
                y_pred_dnn = DNN(X_train, y_train, X_test, y_test)
                
                
                trade_count_lg, winning_ratio_lg, mean_gain_lg , mean_loss_lg, payoff_ratio_lg , sum_gain_lg , sum_loss_lg , profit_factor_lg = pred(test_close, y_pred_lg)
                trade_count_dt, winning_ratio_dt, mean_gain_dt , mean_loss_dt, payoff_ratio_dt , sum_gain_dt , sum_loss_dt , profit_factor_dt = pred(test_close, y_pred_dt)
                trade_count_rf, winning_ratio_rf, mean_gain_rf , mean_loss_rf, payoff_ratio_rf , sum_gain_rf , sum_loss_rf , profit_factor_rf = pred(test_close, y_pred_rf)
                trade_count_xg, winning_ratio_xg, mean_gain_xg , mean_loss_xg, payoff_ratio_xg , sum_gain_xg , sum_loss_xg , profit_factor_xg = pred(test_close, y_pred_xg)
                trade_count_mlp, winning_ratio_mlp, mean_gain_mlp , mean_loss_mlp, payoff_ratio_mlp , sum_gain_mlp , sum_loss_mlp , profit_factor_mlp = pred(test_close, y_pred_mlp)
                trade_count_dnn, winning_ratio_dnn, mean_gain_dnn , mean_loss_dnn, payoff_ratio_dnn , sum_gain_dnn , sum_loss_dnn , profit_factor_dnn = pred(test_close, y_pred_dnn)
                
                
                result_list = []
                    
                result_list.append([i, e, c,"lg", trade_count_lg, winning_ratio_lg, mean_gain_lg , mean_loss_lg, payoff_ratio_lg , sum_gain_lg , sum_loss_lg , profit_factor_lg])
                result_list.append([i, e, c,"dt", trade_count_dt, winning_ratio_dt, mean_gain_dt , mean_loss_dt, payoff_ratio_dt , sum_gain_dt , sum_loss_dt , profit_factor_dt])
                result_list.append([i, e, c,"rf", trade_count_rf, winning_ratio_rf, mean_gain_rf , mean_loss_rf, payoff_ratio_rf , sum_gain_rf , sum_loss_rf , profit_factor_rf])
                result_list.append([i, e, c,"xg", trade_count_xg, winning_ratio_xg, mean_gain_xg , mean_loss_xg, payoff_ratio_xg , sum_gain_xg , sum_loss_xg , profit_factor_xg])
                result_list.append([i, e, c,"mlp", trade_count_mlp, winning_ratio_mlp, mean_gain_mlp , mean_loss_mlp, payoff_ratio_mlp , sum_gain_mlp , sum_loss_mlp , profit_factor_mlp])
                result_list.append([i, e, c,"dnn", trade_count_dnn, winning_ratio_dnn, mean_gain_dnn , mean_loss_dnn, payoff_ratio_dnn , sum_gain_dnn , sum_loss_dnn , profit_factor_dnn])
                
                
                
                df=pd.DataFrame(result_list ,columns=["ticker","epoch", "count","model", "trade_count", "winning_ratio", "mean_gain", "mean_loss", "payoff_ratio" , "sum_gain" , "sum_loss" , "profit_factor"])
                
                print("종목티커 : " , i , "에포크 :", e, "생성개수 : " ,c)
    
                
                result_df = pd.concat([result_df, df])
            
                
            print("epoch :", e)
                                                      
    result_df.to_csv(str(count) + "_monte_result.csv")




# Skrypt modelu LSTM do niezależnego podłączenia 
import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime, timedelta, date
from pathlib import Path
import yfinance as yf
import openpyxl
import pickle
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Definicje ogólne
today = date.today()
comm = 'Silver' #'PLN/GBP' #'EUR/PLN' #'PLN/USD' 
cor_min = 0.05
cor_max = 0.43

comm_dict = {'^GSPC':'SP_500','^DJI':'DJI30','^IXIC':'NASDAQ','000001.SS':'SSE Composite Index','^HSI':'HANG SENG INDEX','^VIX':'CBOE Volatility Index',
             '^RUT':'Russell 2000','^BVSP':'IBOVESPA','^FTSE':'FTSE 100','^GDAXI':'DAX PERFORMANCE-INDEX', '^N100':'Euronext 100 Index','^N225':'Nikkei 225',
             'EURUSD=X':'EUR_USD','EURCHF=X':'EUR_CHF','CNY=X':'USD/CNY', 'GBPUSD=X':'USD_GBP','JPY=X':'USD_JPY','EURPLN=X':'EUR/PLN',
             'PLN=X':'PLN/USD','GBPPLN=X':'PLN/GBP', 'RUB=X':'USD/RUB','DX-Y.NYB':'US Dollar Index','^XDE':'Euro Currency Index', '^XDN':'Japanese Yen Currency Index',
             '^XDA':'Australian Dollar Currency Index','^XDB':'British Pound Currency Index','^FVX':'5_YB','^TNX':'10_YB','^TYX':'30_YB', 
             'CL=F':'Crude_Oil','BZ=F':'Brent_Oil', 'GC=F':'Gold','HG=F':'Copper', 'PL=F':'Platinum','SI=F':'Silver','NG=F':'Natural Gas',
              'ZR=F':'Rice Futures','ZS=F':'Soy Futures','BTC-USD':'Bitcoin USD','ETH-USD':'Ethereum USD'}

def past_data(past):    
    import warnings
    warnings.filterwarnings("ignore")
    m_tab = None  # Inicjalizacja zmiennej m_tab

    for label, name in comm_dict.items():
        col_name = {'Close': name}
        y1 = pd.DataFrame(yf.download(label, start='2003-12-01', end=today, progress=False))[-past:]
        y1.reset_index(inplace=True)
        y11 = y1[['Date','Close']]
        y11.rename(columns=col_name, inplace=True)
        y2 = y1[['Close']]
        y2 = pd.DataFrame(y2.reset_index(drop=True))
        y2.rename(columns=col_name, inplace=True)
        
        if m_tab is None:
            m_tab = y11  # Tworzenie tabeli w pierwszym przebiegu
        else:
            m_tab = pd.concat([m_tab, y2], axis=1)

    m_tab.fillna(0)
    m_tab.to_pickle('past_data.pkl')
      
# Dobór zmiennych - korelacja
def dx_cut_variables(comm, cor_min, cor_max):
    dx_cor = pd.read_pickle('past_data.pkl')
    dx_cut = dx_cor.drop(['Date'], axis=1)
    correlations = dx_cut.corr()
    comm_correlations = correlations[comm]
    filtered_columns = comm_correlations[(comm_correlations >= cor_min) & (comm_correlations <= cor_max)].index
    filtered_dx_cor = dx_cut[[comm] + list(filtered_columns)]
    fil_dx_cor = pd.concat([dx_cor['Date'],filtered_dx_cor], axis=1)
    fil_dx_cor = fil_dx_cor.fillna(0)
    fil_dx_cor.to_pickle('dx_cor.pkl')
    
# Tworzenie dataset do modelu
def data_set_dx(comm):
    dx_df = pd.read_pickle('dx_cor.pkl')
    var_described = dx_df[comm]
    describing_vars = dx_df.drop(['Date',comm], axis=1)
    describing_rr = (describing_vars - describing_vars.shift(1)) / describing_vars.shift(1)  # ta linijka kodu liczy wszystkie stopy zwrotu
    dx_rr_df = pd.concat([var_described, describing_rr], axis=1) #dx_df['Date'],
    dx_rr_f = dx_rr_df.fillna(0)
    dx_rr_f.to_pickle('dx_rr_f.pkl')
            
# Definicje Modelu LSTM
dx_rr_f = pd.read_pickle('dx_rr_f.pkl')
data_set = dx_rr_f
time_step = 200
column_number = dx_rr_f.shape[1]
xdays = 5
epochs_num = 200
applications_days = 200

# Definicja modelu
def LSTM_DX_Model(data_set, time_step, column_number,  xdays, epochs_num, applications_days):  # time_step, column_number, xdays, epochs_num, applications_days
    set_1 = data_set.fillna(0)
    scaler=MinMaxScaler(feature_range=(0,1))
    set_1_scaled = scaler.fit_transform(np.array(set_1)) #set_1_scaled.min(axis=0) #set_1_scaled.max(axis=0)
    set_1_scaled.min(axis=0), set_1_scaled.max(axis=0)
    
    tr_size=int(len(set_1_scaled)*0.7)
    te_size=len(set_1_scaled)-tr_size
    tr_data = set_1_scaled[0:tr_size,:]
    te_data = set_1_scaled[tr_size : tr_size+te_size,:]  # why do we need , : ? [tr_size : tr_size+te_size,:] this cut the last part of array
    def create_dataset(dataset, time_step): # time_step=1
        dataX, dataY = [], []
        for i in range(len(dataset)-time_step-xdays):  # tutaj ustalamy na ile okresów w przód prognozujemy
            a = dataset[i:(i+time_step)]
            dataX.append(a)
            dataY.append(dataset[(i + time_step):(i + time_step + xdays), 0]) # tutaj ustalamy którą zmienną prognozujemy, oraz na ile okresów w przód

        return np.array(dataX), np.array(dataY)
   
    time_step = time_step  # to jest krytyczny parametr
    X_train, y_train = create_dataset(tr_data, time_step)
    X_test, y_test = create_dataset(te_data, time_step)
    model = Sequential()
    model.add(LSTM(time_step,return_sequences=True,input_shape=(time_step, column_number))) # liczba kolumn musi tutaj być taka sama jak w X_train
    model.add(LSTM(time_step,return_sequences=True))
    model.add(LSTM(time_step))
    model.add(Dense(5))
    
    model.compile(loss='mean_squared_error',optimizer='adam')
    print(model.summary())
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs_num, batch_size = 64, verbose = 2) # 250 give better results
    train_data = model.evaluate(X_train, y_train, verbose=0) 
    test_data = model.evaluate(X_test, y_test, verbose=0)
    #print('MAE train_data: %f' % train_data) 
    #print('MAE test_data: %f' % test_data)
    train_predi = model.predict(X_train, verbose=1)
    test_predi = model.predict(X_test, verbose=1)
    set150 = set_1[- applications_days:].to_numpy() #set125 = np.array([set1[-125:].to_numpy()])
    set_150_sc = scaler.transform(set150) #set_125_sc.shape
    set_150_scaled = np.array([set_150_sc]) # set_125_scaled.shape
    forecast = model.predict(set_150_scaled, verbose=1)
    _forecast = (forecast - scaler.min_[0])/scaler.scale_[0]
    _forecast_df = pd.DataFrame(_forecast)
    _forecast_df.to_pickle('_fore_DX.pkl')
       
# Control Panel
past_data(3001)
dx_cut_variables(comm, cor_min, cor_max)
data_set_dx(comm)
LSTM_DX_Model(data_set, time_step, column_number,  xdays, epochs_num, applications_days)

    

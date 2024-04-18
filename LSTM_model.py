from imports import *

os.chdir(os.path.split(os.path.realpath(__file__))[0])
# 读取数据
data0 = pd.read_csv('stl_decomposition.csv')
data0 = data0.iloc[:1000]

window_size = 64
batch_size = 32
epochs = 100
hidden_dim = [32, 16]
train_ratio = 0.8
show_fit = True
show_loss = True
# mode = 'LSTM'  # RNN,GRU,CNN

def set_my_seed():
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(1)
    rn.seed(12345)
    tf.random.set_seed(123)

def evaluation(y_test, y_predict):
    mae = mean_absolute_error(y_test, y_predict)
    mse = mean_squared_error(y_test, y_predict)
    rmse = np.sqrt(mean_squared_error(y_test, y_predict))
    mape = (abs(y_predict - y_test) / y_test).mean()
    r_2 = r2_score(y_test, y_predict)
    return mae, rmse, mape, r_2  # mse

def build_sequences(text, window_size=24):
    x, y = [], []
    for i in range(len(text) - window_size):
        sequence = text[i:i+window_size]
        target = text[i+window_size]
        x.append(sequence)
        y.append(target)
    return np.array(x), np.array(y)

def get_traintest(data, train_size=len(data0), window_size=24):
    train = data[:train_size]
    test = data[train_size - window_size:]
    X_train, y_train = build_sequences(train, window_size=window_size)
    X_test, y_test = build_sequences(test, window_size=window_size)
    return X_train, y_train[:, -1], X_test, y_test[:, -1]

def build_model(X_train, mode, hidden_dim=[32, 16]):
    set_my_seed()
    model = Sequential()
    if mode == 'RNN':
        model.add(SimpleRNN(hidden_dim[0], return_sequences=True, input_shape=(X_train.shape[-2], X_train.shape[-1])))
        model.add(SimpleRNN(hidden_dim[1]))     
    elif mode=='MLP':
        model.add(Dense(hidden_dim[0],activation='relu',input_shape=(X_train.shape[-2],X_train.shape[-1])))
        model.add(Flatten())
        model.add(Dense(hidden_dim[1],activation='relu'))
        
    elif mode=='LSTM':
        # LSTM
        model.add(LSTM(hidden_dim[0],return_sequences=True, input_shape=(X_train.shape[-2],X_train.shape[-1])))
        model.add(LSTM(hidden_dim[1]))
    elif mode=='GRU':
        #GRU
        model.add(GRU(hidden_dim[0],return_sequences=True, input_shape=(X_train.shape[-2],X_train.shape[-1])))
        model.add(GRU(hidden_dim[1]))
    elif mode=='CNN':
        #一维卷积
        model.add(Conv1D(hidden_dim[0], kernel_size=3, padding='causal', strides=1, activation='relu', dilation_rate=1, input_shape=(X_train.shape[-2],X_train.shape[-1])))
        #model.add(MaxPooling1D())
        model.add(Conv1D(hidden_dim[1], kernel_size=3, padding='causal', strides=1, activation='relu', dilation_rate=2))
        #model.add(MaxPooling1D())
        model.add(Flatten())
    model.add(Dense(1))
    model.compile(optimizer='Adam', loss='mse',metrics=[tf.keras.metrics.RootMeanSquaredError(),"mape","mae"])
    return model

# def plot_loss(hist, imfname=''):
#     plt.subplots(1,4,figsize=(16,2))
#     for i,key in enumerate(hist.history.keys()):
#         n=int(str('14')+str(i+1))
#         plt.subplot(n)
#         plt.plot(hist.history[key], 'k', label=f'Training {key}')
#         plt.title(f'{imfname} Training {key}')
#         plt.xlabel('Epochs')
#         plt.ylabel(key)
#         plt.legend()
#     plt.tight_layout()
#     plt.show()

def plot_fit(y_test, y_pred,hist, imfname=''):
    plt.figure(figsize=(4,2))
    plt.plot(y_test, color="red", label="actual")
    plt.plot(y_pred, color="blue", label="predict")
    plt.title(f"拟合值和真实值对比")
    plt.xlabel("Time")
    plt.ylabel('power')
    plt.legend()

    plt.subplots(1,4,figsize=(16,2))
    for i,key in enumerate(hist.history.keys()):
        n=int(str('14')+str(i+1))
        plt.subplot(n)
        plt.plot(hist.history[key], 'k', label=f'Training {key}')
        plt.title(f'{imfname} Training {key}')
        plt.xlabel('Epochs')
        plt.ylabel(key)
        plt.legend()
    plt.tight_layout()

    plt.show()
    
df_eval_all = pd.DataFrame(columns=['MAE', 'RMSE', 'MAPE', 'R2'])
df_preds_all = pd.DataFrame()

# 自定义回调函数
class TrainingProgressCallback(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        print(f"Epoch {epoch+1}/{epochs}")

def train_fuc(mode, epochs, window_size=64, batch_size=32, hidden_dim=[32, 16], train_ratio=0.8, show_loss=True, show_fit=True):
    data = data0.to_numpy()
    scaler = MinMaxScaler() 
    scaler = scaler.fit(data[:, :-1])
    X = scaler.transform(data[:, :-1])   
    y_scaler = MinMaxScaler() 
    y_scaler = y_scaler.fit(data[:, -1].reshape(-1, 1))
    y = y_scaler.transform(data[:, -1].reshape(-1, 1))
    train_size = int(len(data) * train_ratio)
    X_train, y_train, X_test, y_test = get_traintest(np.c_[X, y], window_size=window_size, train_size=train_size)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    s = time.time()
    set_my_seed()
    model = build_model(X_train=X_train, mode=mode, hidden_dim=hidden_dim)
    earlystop = EarlyStopping(monitor='loss', min_delta=0, patience=5)
    progress_callback = TrainingProgressCallback()
    hist = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=[earlystop, progress_callback], verbose=0)
    y_pred = model.predict(X_test)
    y_pred = y_scaler.inverse_transform(y_pred)
    y_test = y_scaler.inverse_transform(y_test.reshape(-1, 1))
    if show_fit:
        plot_fit(y_test, y_pred,hist)
        # plot_loss(hist)

    e = time.time()
    print(f"运行时间为{round(e-s, 3)}")
    df_preds_all[mode] = y_pred.reshape(-1,)
    s = list(evaluation(y_test, y_pred))
    df_eval_all.loc[f'{mode}', :] = s
    s = [round(i, 3) for i in s]
    print(f'{mode}的预测效果为MAE:{s[0]},RMSE:{s[1]},MAPE:{s[2]},R2:{s[3]}')
    print("=======================================运行结束==========================================")


def LSTM_train():
    train_fuc(mode='GRU', window_size=window_size, batch_size=batch_size, epochs=epochs, hidden_dim=hidden_dim, train_ratio=train_ratio, show_loss=show_loss, show_fit=show_fit)

    print("Is GPU available:", tf.test.is_gpu_available())
    print("GPU devices:", tf.config.list_physical_devices('GPU'))

    # 从原始数据的时间戳中选择与预测值数量相匹配的时间戳
    time_index = data0['Time'].iloc[-len(df_preds_all):]

    # 创建一个新的 DataFrame 来保存时间信息和预测值
    df_predicted_with_time = pd.DataFrame(index=time_index)
    df_predicted_with_time['resid'] = df_preds_all['GRU'].values  # 假设 'GRU' 是包含预测值的列

    # 重命名列
    df_predicted_with_time_renamed = df_predicted_with_time.rename(columns={'resid': 'resid'})

    # 将DataFrame保存为CSV文件
    df_predicted_with_time_renamed.to_csv('forecast_resid.csv', index_label='Time')

    # 读取CSV文件
    df = pd.read_csv('forecast_resid.csv')

    # 将时间戳列转换为时间格式
    df['Time'] = pd.to_datetime(df['Time'], unit='s') 

    # 将时区改为北京时间
    beijing_tz = pytz.timezone('Asia/Shanghai') 
    df['Time'] = df['Time'].dt.tz_localize('UTC').dt.tz_convert(beijing_tz)

    # 格式化时间并去除时区信息
    df['Time'] = df['Time'].dt.strftime('%Y-%m-%d %H:%M:%S')

    # 保存修改后的数据框到新的CSV文件
    df.to_csv('forecast_resid.csv', index=False)


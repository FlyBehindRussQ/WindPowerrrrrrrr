from imports import *

def build_models():
    set_seed()
    data.model = Sequential()
    if data.mode==0:
        model_RNN()
    if data.mode==1:
        model_MLP()
    if data.mode==2:
        model_LSTM()
    if data.mode==3:
        model_GRU()
    if data.mode==4:
        model_CNN()
    data.model.add(Dense(1))
    data.model.compile(optimizer='Adam',loss='mse',metrics=[tf.keras.metrics.RootMeanSquaredError(),'mape','mae'])

def model_RNN():
    data.model.add(SimpleRNN(data.hidden_dim[0],return_sequences=True,input_shape=(data.train_x.shape[-2],data.train_x.shape[-1])))
    data.model.add(SimpleRNN(data.hidden_dim[1]))

def model_MLP():
    data.model.add(Dense(data.hidden_dim[0],activation='relu',input_shape=(data.train_x.shape[-2],data.train_x.shape[-1])))
    data.model.add(Flatten())
    data.model.add(Dense(data.hidden_dim[1],activation='relu'))

def model_LSTM():
    data.model.add(LSTM(data.hidden_dim[0],return_sequences=True,input_shape=(data.train_x.shape[-2],data.train_x.shape[-1])))
    data.model.add(LSTM(data.hidden_dim[1]))

def model_GRU():
    data.model.add(GRU(data.hidden_dim[0],return_sequences=True,input_shape=(data.train_x.shape[-2],data.train_x.shape[-1])))
    data.model.add(GRU(data.hidden_dim[1]))

def model_CNN():
    data.model.add(Conv1D(data.hidden_dim[0],kernel_size=3,padding='causal',strides=1,activation='relu',dilation_rate=1,input_shape=(data.train_x.shape[-2],data.train_x.shape[-1])))
    data.model.add(Conv1D(data.hidden_dim[1],kernel_size=3,padding='causal',strides=1,activation='relu',dilation_rate=2))
    data.model.add(Flatten())


def set_seed():
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(1)
    rn.seed(12345)
    tf.random.set_seed(123)
    
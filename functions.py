from drawing import Plots
from model import *
from Combine import *
from LSTM_model import *
from ARIMA_model import *
from regression_model import *
from imports import *

class Functions(Plots):
    def __init__(self, parent=None) -> None:
        # super(Plots,self).__init__(parent)
        self.stage_speed_thresholds = []
        self.stage_speed_custom_range = []
        self.stage_power_custom_range = []
        pass
    
    def check_data(self):
        if data.filepath=='data_after_filter.csv':
            if not os.path.exists('data_after_filter.csv'):
                self.dataState.setText(f'''当前数据更新日期\nNone''')
                self.Train_Model.setText(f'''数据集为空！''')
                self.TrainSetting.setEnabled(False)
                self.Train_Model.setEnabled(False)
                self.dataFilter.setEnabled(False)
                self.dataDBscan.setEnabled(False)
                self.dataDecom.setEnabled(False)
                self.dial('cls')
            else:
                data_version = os.path.getmtime('data_after_filter.csv')
                data_version = datetime.fromtimestamp(int(data_version))
                self.dataState.setText(f'''当前数据更新日期\n{str(data_version)}''')
                self.Train_Model.setText(f'''训练模型''')
                self.TrainSetting.setEnabled(True)
                self.Train_Model.setEnabled(True)
                self.dataFilter.setEnabled(True)
                self.dataDBscan.setEnabled(True)
                self.dataDecom.setEnabled(True)
                self.show_setting()
        else:
            self.dataState.setText(f'''当前数据\n{data.filename}''')
            self.Train_Model.setText(f'''训练模型''')
            self.TrainSetting.setEnabled(True)
            self.Train_Model.setEnabled(True)
            self.dataFilter.setEnabled(False)
            self.dataDBscan.setEnabled(False)
            self.dataDecom.setEnabled(True)
            self.show_setting()
    
    def readln_data(self):
        temp = QtWidgets.QFileDialog.getOpenFileName(self,"选择实际运行数据",r'data.csv',"csv File(*.csv)")
        if temp[0]=='':
            return
        path = temp[0]
        content = pd.read_csv(path)
        content = content[-(content['State']!=10)]
        data.content = content
        data.content.to_csv('data_after_filter.csv',sep=',',index=None)
        data.filepath = 'data_after_filter.csv'
        data.speed = np.array(content['Speed_now'])
        data.power = np.array(content['Power'])
        self.check_data()
        self.plot_data()
    
    def readln_rated(self):
        temp = QtWidgets.QFileDialog.getOpenFileName(self,"选择风机额定数据",r'rated.csv',"csv File(*.csv)")
        if temp[0]=='':
            return
        path = temp[0]
        content = pd.read_csv(path)
        rated_x = content['Speed']
        rated_y = content['Power']
        data.rated_x = np.array(rated_x)
        data.rated_y = np.array(rated_y)
        self.plot_data()
    
    def readln_training(self):
        temp = QtWidgets.QFileDialog.getOpenFileName(self,"选择已清洗后数据",r'testdata.csv',"csv File(*csv)")
        if temp[0]=='':
            return
        path = temp[0]
        name = temp[0].split('/')[-1]
        content = pd.read_csv(path)
        data.filepath = path
        data.filename = name
        data.speed = np.array(content['Speed_now'])
        data.power = np.array(content['Power'])
        self.check_data()
        self.plot_data()

    def filter(self):
        data_content = data.content
        # 使用保存的阶段临界风速和阶段临界功率进行筛选
        data_content = data_content[-(data_content['Power'] < 0)]
        data_content = data_content[-((data_content['Speed_now'] < data.stage_speed_thresholds[0]) & (data_content['Power'] != 0))]
        data_content = data_content[-((data_content['Speed_now'] > data.stage_speed_thresholds[1]) & (data_content['Power'] != 0))]
        data_content = data_content[-((data_content['Speed_now'] > data.stage_speed_custom_ranges[0]) & (data_content['Power'] > data.stage_power_custom_ranges[0]) &
                                      (data_content['Speed_now'] < data.stage_speed_custom_ranges[1]) & (data_content['Power'] < data.stage_power_custom_ranges[1]))]
        data_content = data_content[-((data_content['Speed_now'] > data.stage_speed_custom_ranges[2]) & (data_content['Power'] > data.stage_power_custom_ranges[2]) &
                                      (data_content['Speed_now'] < data.stage_speed_custom_ranges[3]) & (data_content['Power'] < data.stage_power_custom_ranges[3]))]
        if len(data.rated_x) != 0:
            get_rated = interpolate.interp1d(data.rated_x, data.rated_y)
            data_content = data_content[-(data_content['Power'] > 1.2 * get_rated(data_content['Speed_now']))]

        data.content = data_content
        data.content.to_csv('data_after_filter.csv', sep=',', index=None)
        data.filepath = 'data_after_filter.csv'
        data.speed = np.array(data_content['Speed_now'])
        data.power = np.array(data_content['Power'])
        self.check_data()
        self.plot_data()

    
    def dbscan_GPU(self):
        speed = np.array(data.speed,dtype='float32')
        power = np.array(data.power,dtype='float32') / 100
        data_xy = zip(speed,power)
        ###
        dbscan = DBSCAN(eps=0.5,min_samples=300)
        # labels = dbscan.fit_predict(data_xy)
        labels = np.ones(len(speed))
        for i in range(5,500):
            labels[i] = 0
        ###
        data.content.insert(loc=len(data.content.columns),column='labels',value=labels)
        data.content = data.content[(data.content['labels']==1)]
        data.content = data.content.drop(columns='labels')
        data.content.to_csv('data_after_filter.csv',sep=',',index=None)
        data.filepath = 'data_after_filter.csv'
        data.speed = np.array(data.content['Speed_now'])
        data.power = np.array(data.content['Power'])
        self.check_data()
        self.plot_data()
    
    def models_changed(self,index):
        data.mode = index
    
    def train_prepare(self):
        if data.newfunc:
            self.train_models()
        else:
            self.model_training_prepare()

    def model_training_prepare(self):
        all_data = pd.read_csv(data.filepath).iloc[:data.train_size,:]
        if data.filepath=='data_after_filter.csv' and data.dbscan==0:
            all_data = all_data.drop(columns='MachineNo')
            all_data = all_data.drop(columns='Time')
            all_data = all_data.drop(columns='State')
            all_data = all_data.drop(columns='TemperatureInside')
        all_data = all_data.to_numpy()
        
        x_scaler = MinMaxScaler()
        x_scaler = x_scaler.fit(all_data[:,:-1])
        data_x = x_scaler.transform(all_data[:,:-1])
        y_scaler = MinMaxScaler()
        y_scaler = y_scaler.fit(all_data[:,-1].reshape(-1,1))
        data_y = y_scaler.transform(all_data[:,-1].reshape(-1,1))
        data_xy = np.c_[data_x,data_y]
        
        train_size = int(data.train_size*data.train_ratio)
        train_data = data_xy[:train_size]
        test_data = data_xy[train_size-data.window_size:]
        
        train_x,train_y,test_x,test_y = [],[],[],[]
        for i in range(len(train_data)-data.window_size):
            sequence = train_data[i:i+data.window_size]
            target = train_data[i+data.window_size]
            train_x.append(sequence)
            train_y.append(target)
        for i in range(len(test_data)-data.window_size):
            sequence = test_data[i:i+data.window_size]
            target = test_data[i+data.window_size]
            test_x.append(sequence)
            test_y.append(target)

        data.scaler_x = x_scaler
        data.scaler_y = y_scaler
        data.train_x = np.array(train_x)
        data.train_y = np.array(train_y)[:,-1]
        data.test_x = np.array(test_x)
        data.test_y = np.array(test_y)[:,-1]
        data.predict_x = []
        data.predict_y = []
        # print(data.train_x.shape,data.train_y.shape,data.test_x.shape,data.test_y.shape)
        self.model_training()

    def model_training(self):
        start_time = time.time()
        build_models()
        earlystop = EarlyStopping(monitor='loss',min_delta=0,patience=5)
        callbacks = TrainingProgressCallback()
        hist = data.model.fit(data.train_x,data.train_y,batch_size=data.batch_size,epochs=data.epochs,callbacks=[earlystop,callbacks],verbose=0)
        data.hist = hist
        predicts = data.model.predict(data.test_x)
        data.predicts = data.scaler_y.inverse_transform(predicts)
        data.test_y = data.scaler_y.inverse_transform(data.test_y.reshape(-1,1))
        self.plot_model()
        
        end_time = time.time()
        run_time = round(end_time-start_time,3)
        
        predicts_all = pd.DataFrame()
        predicts_all[data.mode] = data.predicts.reshape(-1,)
        data.mae = round(mean_absolute_error(data.test_y,data.predicts),3)
        data.mse = round(mean_squared_error(data.test_y,data.predicts),3)
        data.rmse = round(np.sqrt(data.mse),3)
        count = 0
        for i in range(len(data.test_y)):
            if data.test_y[i]!=0:
                data.mape = data.mape + float(abs(data.predicts[i] - data.test_y[i]) / data.test_y[i])
                count = count + 1
        data.mape = round(data.mape / count,3)
        data.r_2 = round(r2_score(data.test_y,data.predicts),3)
        data.runtime = run_time
        self.plot_error()
        
        self.show_results()
        
        print(f'''运行时间为{data.runtime}s''')
        print(f'{data.mode_list[data.mode]}的预测效果为：\tMAE:{data.mae} \tRMSE:{data.rmse} \tMAPE:{data.mape} \tR2:{data.r_2}')
        print("====================运行结束====================")
        print("Is GPU available:", tf.test.is_gpu_available())
        print("GPU devices:", tf.config.list_physical_devices('GPU'))
        
    
    def fake_predict(self):
        temp = QtWidgets.QFileDialog.getOpenFileName(self,"选择预测数据",r'predictdata.csv',"csv File(*.csv)")
        if temp[0]=='':
            return
        path = temp[0]
        content = pd.read_csv(path)
        content = content.to_numpy()
        x_scaler = MinMaxScaler()
        x_scaler = x_scaler.fit(content[:,:-1])
        data_x = x_scaler.transform(content[:,:-1])
        y_scaler = MinMaxScaler()
        y_scaler = y_scaler.fit(content[:,-1].reshape(-1,1))
        data_y = y_scaler.transform(content[:,-1].reshape(-1,1))
        data_xy = np.c_[data_x,data_y]
        
        predict_x,predict_y = [],[]
        for i in range(len(data_xy)-data.window_size):
            sequence = data_xy[i:i+data.window_size]
            predict_x.append(sequence)
        data.predict_x = np.array(predict_x)
        predict_y = data.model.predict(data.predict_x)
        data.predict_y = data.scaler_y.inverse_transform(predict_y)
        self.plot_model()
        
    def fake_dbscan(self):
        content = pd.read_csv('testdata.csv')
        data.content = content
        data.content.to_csv('data_after_filter.csv',sep=',',index=None)
        data.speed = np.array(content['Speed_now'])
        data.power = np.array(content['Power'])
        data.dbscan = 1
        self.check_data()
        self.plot_data()

    def decom_data(self):
        data.filepath = "10yuan_data.csv"
        df = pd.read_csv(data.filepath)
        start_time = datetime(2022, 1, 1, 0, 0)
        time_increment = timedelta(minutes=30)
        df['Time'] = [(start_time + i * time_increment).strftime('%Y-%m-%d %H:%M') for i in range(len(df))]
        df.to_csv(data.filepath, index=False)
 
        data.res = STL(df['power'], period=24*6*30).fit()  #period确定：10分钟一个数据，以天为周期就是24*6
        self.plot_decom()

        df['trend']=data.res.trend
        df['seasonal']=data.res.seasonal
        df['resid']=data.res.resid

        data.filepath = "stl_decomposition.csv"
        df.to_csv(data.filepath)
        data.newfunc = True

        trend_strength=max(0,1-round(df.resid.var()/df.seasonal.var(),3))
        seasonal_strength=max(0,1-round(df.resid.var()/df.trend.var(),3))
        residual_mean=round(df.resid.mean(),3)
        print('trend_strength:',trend_strength)
        print('seasonal_strength:',seasonal_strength)
        print('residual mean:',residual_mean)
        df.resid.hist()
    
    def train_models():

        LSTM_train()
        regression_train()
        ARIMA_train()

        combination()


class TrainingProgressCallback(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        # Imports.dial(data.ui,f"Epoch {epoch+1}/{data.epochs}")
        print(f"Epoch {epoch+1}/{data.epochs}")
        
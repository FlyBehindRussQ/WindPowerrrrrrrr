from windowchild import Windows
from functions import Functions
from drawing import Plots
import data

class MyMainWindow(Windows,Functions,Plots):
    def __init__(self, parent=None) -> None:
        super(Functions,self).__init__(parent)
        super(Windows,self).__init__(parent)
        super(Plots,self).__init__(parent)
        self.Init()
        self.Signals()
        pass
    
    def Signals(self):
        self.ImportData.clicked.connect(self.readln_data)
        self.ImportRated.clicked.connect(self.readln_rated)
        self.ImportTraining.clicked.connect(self.readln_training)
        self.dataFilter.clicked.connect(self.filter_setting_window)
        self.dataDBscan.clicked.connect(self.fake_dbscan)
        self.dataDecom.clicked.connect(self.decom_data)
        self.Train_Model.clicked.connect(self.train_prepare)
        self.TrainSetting.clicked.connect(self.train_setting_window)
        # self.Predict.clicked.connect(self.predict_setting_window)
        self.Predict.clicked.connect(self.fake_predict)
        self.plotshow.clicked.connect(self.plot_show)
        
    def Init(self):
        self.plot_decom()
        self.plot_data()
        self.plot_model()
        self.plot_error()
        self.check_data()
        # self.Models.addItems(data.mode_list)
        # self.Models.setCurrentIndex(data.mode)
        self.Viewer.setCurrentIndex(0)
        
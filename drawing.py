from imports import *

class Plots(Imports):
    def __init__(self, parent=None) -> None:
        # super(Imports,self).__init__(parent)
        pass
    
    def plot_data(self):
        self.layout_data = QtWidgets.QGridLayout()
        self.view_data = QtWidgets.QLabel(self.Tab_Data)
        self.layout_data.addWidget(self.view_data,0,0,1,1)
        self.Layout_DataView.addLayout(self.layout_data,0,0,1,1)
        try:
            sip.delete(data.dataPlot)
            sip.delete(data.dataPlot_Toolbar)
        except:
            pass
        self.view_data.hide()
        data.dataPlot = MyFigure(view=self.view_data)
        data.dataPlot_Toolbar = NavigaionToolbar(data.dataPlot,self)
        data.dataPlot.axes = data.dataPlot.fig.add_subplot(111)
        data.dataPlot.axes.set_title("数据集")
        data.dataPlot.axes.set_xlabel("风速(m/s)")
        data.dataPlot.axes.set_ylabel("功率(W)")
        if len(data.rated_x)!=0:
            data.dataPlot.axes.plot(data.rated_x,data.rated_y,color='b',label="额定功率曲线")
            data.dataPlot.axes.legend()
        if len(data.speed)!=0:
            data.dataPlot.axes.scatter(data.speed,data.power,c='r',s=1)
        self.layout_data.addWidget(data.dataPlot,0,1,3,1)
        self.layout_data.addWidget(data.dataPlot_Toolbar,3,1)

    def plot_decom(self):
        self.layout_decom = QtWidgets.QGridLayout()
        self.view_decom = QtWidgets.QLabel(self.Tab_Decom)
        self.layout_decom.addWidget(self.view_decom,0,0,1,1)
        self.Layout_DecomView.addLayout(self.layout_decom,0,0,1,1)
        try:
            sip.delete(data.decomPlot)
            sip.delete(data.decomPlot_Toolbar)
        except:
            pass
        self.view_decom.hide()
        data.decomPlot = MyFigure(view=self.view_decom)
        data.decomPlot_Toolbar = NavigaionToolbar(data.decomPlot,self)
        data.decomPlot.axes = data.decomPlot.fig.add_subplot(111)
        if data.res!=None:
            data.res.plot()
        self.layout_decom.addWidget(data.decomPlot,0,1,3,1)
        self.layout_decom.addWidget(data.decomPlot_Toolbar,3,1)

        sip.delete(data.decomPlot)
        sip.delete(data.decomPlot_Toolbar)
        data.decomPlot = MyFigure(view=self.view_decom)
        data.decomPlot_Toolbar = NavigaionToolbar(data.decomPlot,self)
        data.decomPlot.axes = data.decomPlot.fig.add_subplot(111)
        data.decomPlot.axes.set_title("数据分解")
        if data.res!=None:
            data.decomPlot.plot(data.res)
        self.layout_decom.addWidget(data.decomPlot,0,1,3,1)
        self.layout_decom.addWidget(data.decomPlot_Toolbar,3,1)

    
    def plot_model(self):
        self.layout_model = QtWidgets.QGridLayout()
        self.view_model = QtWidgets.QLabel(self.Tab_Model)
        self.layout_model.addWidget(self.view_model,0,0,1,1)
        self.Layout_ModelView.addLayout(self.layout_model,0,0,1,1)
        try:
            sip.delete(data.modelPlot)
            sip.delete(data.modelPlot_Toolbar)
        except:
            pass
        self.view_model.hide()
        data.modelPlot = MyFigure(view=self.view_model)
        data.modelPlot_Toolbar = NavigaionToolbar(data.modelPlot,self)
        data.modelPlot.axes = data.modelPlot.fig.add_subplot(111)
        data.modelPlot.axes.set_title("模型训练结果")
        data.modelPlot.axes.set_xlabel("数据量")
        data.modelPlot.axes.set_ylabel("功率(W)")
        if len(data.test_y)!=0:
            if len(data.predict_y)!=0:
                data.modelPlot.axes.set_title("模型训练结果+预测结果")
                data.predict_y = np.r_[data.predicts,data.predict_y]
                data.modelPlot.axes.plot(data.predict_y,color='g',label="预测值")
            data.modelPlot.axes.plot(data.test_y,color='r',label="实际值")
            data.modelPlot.axes.plot(data.predicts,color='b',label="训练值")
            data.modelPlot.axes.legend()
        self.layout_model.addWidget(data.modelPlot,0,1,3,1)
        self.layout_model.addWidget(data.modelPlot_Toolbar,3,1)
        self.Viewer.setCurrentIndex(2)
    
    def plot_error(self):
        self.layout_error = QtWidgets.QGridLayout()
        self.view_error = QtWidgets.QLabel(self.Tab_Error)
        self.layout_error.addWidget(self.view_error,0,0,1,1)
        self.Layout_ErrorView.addLayout(self.layout_error,0,0,1,1)
        try:
            sip.delete(data.errorPlot)
        except:
            pass
        self.view_error.hide()
        data.errorPlot = MyFigure(view=self.view_error)
        if data.hist==None:
            keys = ['loss','squared_error','mape','mae']
            for key in keys:
                data.errorPlot.axes = data.errorPlot.fig.add_subplot(int(f'22{keys.index(key)+1}'))
                data.errorPlot.axes.set_title(f'Training {key}')
                data.errorPlot.axes.set_xlabel('Epochs')
                data.errorPlot.axes.set_ylabel(key)
        else:
            keys = ['loss','squared_error','mape','mae']
            for i,key in enumerate(data.hist.history.keys()):
                data.errorPlot.axes = data.errorPlot.fig.add_subplot(int(f'22{i+1}'))
                data.errorPlot.axes.set_title(f'Training {keys[i]}')
                data.errorPlot.axes.set_xlabel('Epochs')
                data.errorPlot.axes.set_ylabel(keys[i])
                data.errorPlot.axes.plot(data.hist.history[key], 'k')
        self.layout_error.addWidget(data.errorPlot,0,0)
        
    def plot_show(self):
        self.layout_data = QtWidgets.QGridLayout()
        self.view_data = QtWidgets.QLabel(self.Tab_Data)
        self.layout_data.addWidget(self.view_data,0,0,1,1)
        self.Layout_DataView.addLayout(self.layout_data,0,0,1,1)
        try:
            sip.delete(data.dataPlot)
            sip.delete(data.dataPlot_Toolbar)
        except:
            pass
        self.view_data.hide()
        img = Image("stl.png")
        data.dataPlot = MyFigure(view=self.view_data)
        data.dataPlot_Toolbar = NavigaionToolbar(data.dataPlot,self)
        data.dataPlot.axes = data.dataPlot.fig.add_subplot(111)
        data.dataPlot.axes.imshow(img)
        data.dataPlot.axes.set_title("数据集")
        data.dataPlot.axes.set_xlabel("风速(m/s)")
        data.dataPlot.axes.set_ylabel("功率(W)")
        self.layout_data.addWidget(data.dataPlot,0,1,3,1)
        self.layout_data.addWidget(data.dataPlot_Toolbar,3,1)
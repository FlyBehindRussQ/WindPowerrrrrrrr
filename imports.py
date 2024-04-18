import os,sys
import data

from PyQt5 import QtWidgets,sip
from PyQt5.QtGui import QFont,QTextDocument
from PyQt5.QtCore import Qt, QCoreApplication, QRect
from WindowGUI import Ui_MainWindow

import math
import pytz
import time
import numpy as np
import pandas as pd
import random as rn
from scipy import interpolate,optimize
from datetime import datetime,timedelta

import statsmodels.api as sm
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX

from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error,r2_score

import tensorflow as tf
from keras.models import Model,Sequential
from keras.layers import LSTM,Dense,Conv1D,Flatten,SimpleRNN,GRU
from keras.callbacks import EarlyStopping,Callback

from PIL import Image

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigaionToolbar


class Imports(QtWidgets.QMainWindow, Ui_MainWindow):
    def __del__(self):
        pass
    
    def __init__(self, parent=None) -> None:
        super(Imports, self).__init__(parent)
        self.setupUi(self)
        pass
    
    def dial(self,text):
        if text=='cls':
            self.dialogue.clear()
            return
        str=f'''<p style="color:rgb(0,0,0)">{text}</p>'''
        self.dialogue.append(str)
        
    def show_setting(self):
        self.dial('cls')
        self.dial(f'''训练模型: {data.mode_list[data.mode]}''')
        self.dial(f'''训练次数: {data.epochs}''')
        self.dial(f'''模型层数: {data.hidden_dim}''')
        self.dial(f'''数据量: {data.train_size}''')
        self.dial(f'''训练集: {int(data.train_size * data.train_ratio)}''')
        
    def show_results(self):
        self.dial('cls')
        self.dial(f'''{data.mode_list[data.mode]}运行时间: {data.runtime}s''')
        self.dial(f'''MAE: {data.mae}''')
        self.dial(f'''RMSE: {data.rmse}''')
        self.dial(f'''MAPE: {data.mape}''')
        self.dial(f'''R2: {data.r_2}''')


class MyFigure(FigureCanvas):
    def __init__(self,view=None):
        self.cb = None
        self.fig = Figure(figsize=(100, 100), dpi=100)
        super(MyFigure,self).__init__(self.fig)
        # self.axes = self.fig.add_subplot(1,1,1)
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['figure.constrained_layout.use'] = True
        
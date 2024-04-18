#####################
## GUI
#####################
ui = None

dataPlot = None
decomPlot = None
modelPlot = None
errorPlot = None
dataPlot_Toolbar = None
modelPlot_Toolbar = None

#####################
## DATA
#####################
rated_x = []
rated_y = []

content = None
speed = []
power = []

dbscan = 0
#####################
## MODEL
#####################
filepath = 'data_after_filter.csv'
filename = ''

mode = 2
mode_list = ['RNN','MLP','LSTM','GRU','CNN']

model = None

train_size = 1000
train_ratio = 0.8
window_size = 64
batch_size = 32
epochs = 100
hidden_dim = [32,16]

scaler_x = None
scaler_y = None

train_x = []
train_y = []

test_x = []
test_y = []

predict_x = []
predict_y = []

hist = None
predicts = []

mae = 0
mse = 0
rmse = 0
mape = 0
r_2 = 0
runtime = 0


#####################
## PREDICTION
#####################

speed_now = 0
speed_sec = 0
speed_min = 0
direction = 0
temperature = 0
pitch_angle = 0

input_x = []
output_y = []


#####################
## FILTER
#####################

stage_speed_thresholds = [3,20]
stage_speed_custom_ranges = [10,20,13,20]
stage_power_custom_ranges = [900,1100,3500,3640]


#####################
## SOMETHING NEW
#####################
res = None
newfunc = False
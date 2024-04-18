from imports import *

def regression_train(self):
    data = pd.read_csv('stl_decomposition.csv')

    # 将时间列转换为时间戳类型
    data['Time'] = pd.to_datetime(data['Time'])

    # # 定义二次函数
    def function(x, a, b, c):
        return a * x**2 + b * x + c

    # 提取趋势数据
    x_data = np.arange(len(data))
    y_data = data['trend']

    # 使用 curve_fit 函数拟合趋势数据
    popt, pcov = optimize.curve_fit(function, x_data, y_data)

    # 提取拟合的参数值
    a_fit, b_fit, c_fit = popt

    # 绘制原始数据和拟合结果
    regression = function(x_data, a_fit, b_fit, c_fit)

    plt.scatter(x_data, y_data, label='Original Data')
    plt.plot(x_data, regression, color='red', label='Fitted Curve')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Nonlinear Regression of Trend Data using Quadratic Function')
    plt.show()

    # 打印拟合的参数值
    print("Fitted Parameters:")
    print("a =", a_fit)
    print("b =", b_fit)
    print("c =", c_fit)

    # 将时间和回归值合并成一个DataFrame
    regression_data = pd.DataFrame({'Time': data['Time'], 'Regression': regression})

    # 保存数据到CSV文件
    regression_data.to_csv('forecast_trend.csv', index=False)

    # 计算 R^2 分数
    r2 = r2_score(y_data, regression)

    print("R^2 Score:", r2)
    
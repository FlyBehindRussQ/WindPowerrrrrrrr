from imports import *

def ARIMA_train():
    # 读取CSV文件
    data = pd.read_csv('stl_decomposition.csv')

    # data = data[:300]

    # 将时间列转换为时间戳类型
    data['Time'] = pd.to_datetime(data['Time'])

    # 设置时间列为索引
    data.set_index('Time', inplace=True)

    # 绘制ACF图
    fig, ax = plt.subplots(figsize=(12, 6))
    sm.graphics.tsa.plot_acf(data['seasonal'], ax=ax, lags=30)
    plt.title('Autocorrelation Function (ACF)')
    plt.xlabel('Lag')
    plt.ylabel('ACF')
    plt.show()

    # 绘制PACF图
    fig, ax = plt.subplots(figsize=(12, 6))
    sm.graphics.tsa.plot_pacf(data['seasonal'], ax=ax, lags=30)
    plt.title('Partial Autocorrelation Function (PACF)')
    plt.xlabel('Lag')
    plt.ylabel('PACF')
    plt.show()

    train_size = int(len(data) * 0.8)

    # 初始化预测列表
    all_forecast_values = []

    # 滑动窗口进行多次的单步预测
    for i in range(train_size, len(data)):
        train_data = data[:i]  # 根据当前滑动窗口的索引更新训练集
        test_data = data.iloc[i:i+1]  # 每次只预测一个时间步
        
        # SARIMAX模型参数设置
        order = (1, 0, 1)  # ARIMA(p,d,q) 参数
        seasonal_order = (1, 0, 1, 12)  # 季节性部分参数

        # 创建并拟合SARIMAX模型
        model = SARIMAX(train_data['seasonal'], order=order, seasonal_order=seasonal_order)
        results = model.fit()

        # 对测试集进行预测
        forecast = results.get_forecast(steps=1)  # 每次只预测一个时间步

        # 获取预测结果
        forecast_values = forecast.predicted_mean.values[0]

        # 添加当前预测结果到预测列表中
        all_forecast_values.append(forecast_values)

    # 创建索引以匹配测试集
    forecast_index = data.index[train_size:]

    # 可视化预测结果
    plt.figure(figsize=(12, 6))
    plt.plot(data.index[:train_size], data['seasonal'][:train_size], label='Training Data')
    plt.plot(data.index[train_size:], data['seasonal'][train_size:], label='Test Data')
    plt.plot(forecast_index, all_forecast_values, label='Forecast')
    plt.xlabel('Time')
    plt.ylabel('seasonal power')
    plt.title('Wind Power Forecast using SARIMAX with Sliding Window')
    plt.legend()
    plt.show()

    # 将预测结果转换为DataFrame
    forecast_df = pd.DataFrame({'Time': forecast_index, 'Seasonal': all_forecast_values})

    # 删除第一列
    # forecast_df = forecast_df.drop(forecast_df.columns[0], axis=1)

    # 将预测结果写入CSV文件
    forecast_df.to_csv('forecast_seasonal.csv', index=True)

    # 计算 RMSE
    rmse = np.sqrt(mean_squared_error(data['seasonal'][train_size:], all_forecast_values))

    # 计算 MAE
    mae = mean_absolute_error(data['seasonal'][train_size:], all_forecast_values)

    # 计算 MSE
    mse = mean_squared_error(data['seasonal'][train_size:], all_forecast_values)

    # 计算 MAPE
    mape = np.mean(np.abs((data['seasonal'][train_size:] - all_forecast_values) / data['seasonal'][train_size:])) * 100

    # 计算 R^2
    r2 = r2_score(data['seasonal'][train_size:], all_forecast_values)

    print('RMSE',rmse)
    print('MAE',mae)
    print('MSE',mse)
    print('MAPE',mape)
    print('R2',r2)

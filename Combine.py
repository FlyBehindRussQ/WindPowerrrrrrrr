from imports import *

def combination():
    original = pd.read_csv('stl_decomposition.csv')
    # original = original.iloc[800:1000]

    original_power = original['power']

    forecast_trend = pd.read_csv('forecast_trend.csv')
    forecast_seasonal = pd.read_csv('forecast_seasonal.csv')
    forecast_resid = pd.read_csv('forecast_resid.csv')

    # 合并预测值数据并设置时间戳列为索引
    merged_forecast = pd.merge(forecast_trend, forecast_seasonal, on='Time')
    merged_forecast = pd.merge(merged_forecast, forecast_resid, on='Time')

    # 将缺失值填充为0（如果有的话）
    merged_forecast.fillna(0, inplace=True)

    # 执行相加操作
    merged_forecast['Combined'] = merged_forecast['Regression'] + merged_forecast['Seasonal'] + merged_forecast['resid']

    print(merged_forecast)
    merged_forecast.to_csv('forecast_combined.csv')

    # 提取指定行范围的数据
    subset = original.iloc[13156:16001]

    # 重新设置索引，使其从0开始
    subset.reset_index(drop=True, inplace=True)

    plt.figure(figsize=(12,8))
    plt.plot(subset['power'],color = 'blue',label = 'Original')
    plt.plot(merged_forecast['Combined'], color='red', label='Forecast')
    plt.xlabel('Index')
    plt.ylabel('Power')
    plt.title('Original Data vs Combined Forcast Data')
    plt.legend()
    plt.show()


    # 计算 RMSE
    rmse = np.sqrt(mean_squared_error(subset['power'], merged_forecast['Combined'][:2845]))

    # 计算 MAE
    mae = mean_absolute_error(subset['power'], merged_forecast['Combined'][:2845])

    # 计算 MSE
    mse = mean_squared_error(subset['power'], merged_forecast['Combined'][:2845])

    # 计算 MAPE
    mape = np.mean(np.abs((subset['power'] - merged_forecast['Combined'][:2845]) / subset['power'])) * 100

    # 计算 R^2
    r2 = r2_score(subset['power'], merged_forecast['Combined'][:2845])

    print('RMSE',rmse)
    print('MAE',mae)
    print('MSE',mse)
    print('MAPE',mape)
    print('R2',r2)

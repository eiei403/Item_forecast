import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.callbacks import EarlyStopping
from xgboost import XGBRegressor
from loadandcleandata import load
from xgboost import plot_importance
from pandas.tseries.offsets import MonthEnd

from scipy.stats import percentileofscore

def quantile_match(forecast_values, reference_values):
    quantiles = [percentileofscore(forecast_values, val) for val in forecast_values]
    return np.percentile(reference_values, quantiles)

def forecast_2025():
    df = load()
    #df = df.groupby(df['Date'].dt.to_period("M")).sum(numeric_only=True).reset_index()
    df = df[(df['Date'] >= '2022-01-01') & (df['Date'] <= '2025-05-31')]
    df.sort_values('Date', inplace=True)

    df['Lag1'] = df['Quantity'].shift(1)
    df['Lag2'] = df['Quantity'].shift(2)
    df['Lag3'] = df['Quantity'].shift(3)
    df['Month'] = df['Date'].dt.month
    df['year'] = df['Date'].dt.year
    df['diff1'] = df['Quantity'].diff(1)
    df['diff3'] = df['Quantity'].diff(3)
    #df['IsMonthEndDip'] = df['Month'].apply(lambda x: 1 if x in [4, 12] else 0)  # เดือนที่เคยร่วงแรง
    df['rolling_mean_3'] = df['Quantity'].rolling(window=3).mean()
    df['month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    #df['IsMonthEndDip_Lag1'] = df['IsMonthEndDip'] * df['Lag1']
    df['Lag_Diff'] = df['Lag1'] - df['Lag2']
    # df['trend_slope_6m'] = (
    #     df['Quantity'].transform(lambda x: x.shift(1).rolling(6).apply(
    #         lambda series: np.polyfit(np.arange(len(series)), series, deg=1)[0]
    #         if len(series) >= 2 else 0.0, raw=False
    #     ))
    # )
    #df['IsMonthEndDip_trend'] = df['IsMonthEndDip'] * df['trend_slope_6m']
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    #df['IsYearEnd'] = df['Month'].apply(lambda x: 1 if x == 12 else 0)
    #df['IsQuarterEnd'] = df['Month'].apply(lambda x: 1 if x in [3, 6, 9, 12] else 0)
    df['IsApril'] = df['Month'].apply(lambda x: 1 if x == 4 else 0)
    df['residual_lag1'] = df['Quantity'] - df['Lag1']  # ใช้แทน diff
    df['rolling_std_3'] = df['Quantity'].rolling(window=3).std()    
    df['rolling_mean_6'] = df['Quantity'].rolling(window=6).mean()
    df['rolling_diff_3'] = df['Quantity'] - df['rolling_mean_3']
    #df['IsMonthEndDip_trend'] = df['IsMonthEndDip'] * df['trend_slope_6m']
    df['April_Drop'] = df['Month'].apply(lambda x: 1 if x == 4 else 0) * df['diff1']
    df['Lag1_vs_Mean6'] = df['Lag1'] / (df['rolling_mean_6'] + 1e-5)
    #df['residual_trend'] = df['Quantity'] - (df['trend_slope_6m'] * np.arange(len(df)))
    df['IsFalling'] = df['diff1'].apply(lambda x: 1 if x < 0 else 0)
    #df['mean_diff_ratio'] = df['diff1'] / (df['rolling_mean_6'] + 1e-5)
    #df['lag1_resid_to_mean'] = (df['residual_lag1']) / (df['rolling_mean_6'] + 1e-5)
    #df['slope_sign'] = df['trend_slope_6m'].apply(lambda x: 1 if x > 0 else -1)
    df.dropna(inplace=True)

    features = [
        'rolling_diff_3','diff3','rolling_mean_3','residual_lag1','diff1','rolling_mean_6',
        'rolling_std_3','Lag1_vs_Mean6','day_cos'
        ,'IsFalling','Lag1','Lag2','Lag3','day_sin'
    ]
    #,,'IsMonthEndDip_trend','April_Drop','residual_trend','Lag_Diff'
    X = df[features]
    y = df['Quantity']
    df.to_csv('data.csv',index = False)

    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()

    split = int(len(df) * 0.95)
    X_train, y_train = X.iloc[:split], y_scaled[:split]
    X_test, y_test = X.iloc[split:], y_scaled[split:]

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=3000,
        learning_rate=0.01,   # ลด learning rate
        max_depth=5,          # ลดความซับซ้อนของแต่ละต้น
        subsample=0.6,
        colsample_bytree=0.6,
        min_child_weight=8,
        reg_alpha=1.0,
        reg_lambda=1.0,
        n_jobs=1,
        random_state=1301
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='rmse',
        verbose=True,
        early_stopping_rounds=50
    )

    residuals = df['Quantity'] - df['rolling_mean_3'] 
    noise_std = residuals.std() 
    
    last_df = df.copy().reset_index(drop=True)
    forecast_days = pd.date_range(start='2025-06-01', end='2025-08-31', freq='D')
    
    future_predictions = []
    np.random.seed(42)  

    for i, forecast_date in enumerate(forecast_days):
        last_row = last_df.iloc[-1]
        
        new_row = {
            'Date': forecast_date,
            'Lag1': last_row['Quantity'],
            'Lag2': last_df.iloc[-2]['Quantity'],
            'Lag3': last_df.iloc[-3]['Quantity'],
        }
        
        new_row['diff1'] = new_row['Lag1'] - new_row['Lag2']
        new_row['diff3'] = new_row['Lag1'] - last_df.iloc[-4]['Quantity']
        new_row['rolling_mean_3'] = last_df['Quantity'].iloc[-3:].mean()
        new_row['rolling_std_3'] = last_df['Quantity'].iloc[-3:].std()
        new_row['rolling_mean_6'] = last_df['Quantity'].iloc[-6:].mean()
        new_row['rolling_diff_3'] = new_row['Lag1'] - new_row['rolling_mean_3']
        new_row['Lag1_vs_Mean6'] = new_row['Lag1'] / (new_row['rolling_mean_6'] + 1e-5)
        new_row['residual_lag1'] = new_row['Lag1'] - new_row['rolling_mean_3']
        new_row['IsFalling'] = 1 if new_row['diff1'] < 0 else 0
        
        dow = forecast_date.dayofweek
        new_row['day_sin'] = np.sin(2 * np.pi * dow / 7)
        new_row['day_cos'] = np.cos(2 * np.pi * dow / 7)
        
        new_df = pd.DataFrame([new_row])
        X_future = new_df[features]
        y_future_scaled = model.predict(X_future)
        y_future_base = scaler_y.inverse_transform(y_future_scaled.reshape(-1, 1)).ravel()[0]
        
        # เพิ่ม noise และ seasonality
        noise = np.random.normal(0, noise_std * 0.4)  # ลด noise ลงเล็กน้อย
        
        weekly_effect = np.sin(2 * np.pi * dow / 7) * (noise_std * 0.2)
        
        # decay factor เพื่อลดความผันผวนตามเวลา
        decay_factor = max(0.5, 1 - (i / len(forecast_days)) * 0.3)
        
        y_future = y_future_base + (noise + weekly_effect) * decay_factor
        
        # ป้องกันค่าติดลบ
        y_future = max(y_future, 0)
        
        new_row['Quantity'] = y_future
        future_predictions.append(new_row)
        
        last_df = pd.concat([last_df, pd.DataFrame([new_row])], ignore_index=True)

    y_pred_scaled = model.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
   

    fig, ax = plt.subplots(figsize=(14, 6))
    x_values = df['Date'].iloc[split:]
    x_train_values = df['Date'].iloc[:split]
    result = pd.DataFrame({
        'date': x_values,
        'y_pred': y_pred,
        'y_actual': y_test_actual
    })
    result.to_csv('./forecast9896/result.csv',index=False)
    # plot_importance(model, max_num_features=20)
    # plt.tight_layout()
    # plt.show()
    future_df = pd.DataFrame(future_predictions)
    future = pd.DataFrame({
        'date': future_df['Date'],
        'y': future_df['Quantity']
    })
    future.to_csv('./forecast9896/forecast.csv',index=False)
    # plt.figure(figsize=(14, 6))
    # plt.plot(df['Date'], df['Quantity'], label='Historical')
    # plt.title("Forecast Quantity Until Dec 2025")
    # plt.xlabel("Month")
    # plt.ylabel("Quantity")
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    #ax.plot(x_train_values, scaler_y.inverse_transform(y_train.reshape(-1, 1)), label='Actual (train)', color='red', alpha=0.5)
    ax.plot(x_values, y_test_actual, label='Actual (Test)', color='black', alpha=0.5)
    ax.plot(x_values, y_pred, label='Predicted (Test)', color='blue', marker='o')
    ax.plot(future_df['Date'], future_df['Quantity'], label='Forecast Daily (Jun-Aug)', marker='o', markersize=3,alpha=0.5)
    ax.set_title("Forecast Quantity Until Dec 2025")
    ax.set_xlabel('Date')
    ax.set_ylabel('Quantity')
    ax.legend()
    ax.grid(True)
    fig.autofmt_xdate()
    fig.tight_layout()
    plt.show()

    print("MSE:", mean_squared_error(y_test_actual, y_pred))
    print("MAE:", mean_absolute_error(y_test_actual, y_pred))
    print("R²:", r2_score(y_test_actual, y_pred))
    y_train_pred = model.predict(X_train)
    r2_train = r2_score(y_train, y_train_pred)
    print("Train R²:", r2_train)


forecast_2025()

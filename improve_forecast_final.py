import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor,early_stopping

import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')
from loadandcleandata import load
import math
from pandas.tseries.offsets import DateOffset


class MultiModelForecaster:
    def __init__(self):
        self.models = {
            'xgb': XGBRegressor(
                objective="reg:squarederror", 
                n_estimators=2000, 
                learning_rate=0.02, 
                max_depth=4,
                subsample=0.7, 
                colsample_bytree=0.7, 
                min_child_weight=5, 
                reg_alpha=0.5, 
                reg_lambda=0.5,
                random_state=42
            ),
            'lgb': LGBMRegressor(
                objective='regression',
                n_estimators=2000,
                learning_rate=0.02,
                num_leaves=31,
                feature_fraction=0.7,
                bagging_fraction=0.7,
                random_state=42,
                verbose=-1
            ),
            'rf': RandomForestRegressor(
                n_estimators=500,
                max_depth=8,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        }
        self.scaler_y = StandardScaler()
        self.feature_importance = {}
        self.model_weights = {}
        
    def create_features(self, df):
        """Enhanced feature engineering with better handling of lags"""
        df = df.copy()
        df.sort_values('Date', inplace=True)
        
        # Basic time features
        df['YearMonth'] = df['Date'].dt.to_period('M')
        df['Month'] = df['Date'].dt.month
        df['Year'] = df['Date'].dt.year
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['DayOfMonth'] = df['Date'].dt.day
        df['Quarter'] = df['Date'].dt.quarter
        
        # Cyclical encoding
        df['month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
        df['quarter_sin'] = np.sin(2 * np.pi * df['Quarter'] / 4)
        df['quarter_cos'] = np.cos(2 * np.pi * df['Quarter'] / 4)
        
        # Lag features (more robust)
        for lag in [1, 2, 3, 7, 14, 30]:
            df[f'Lag_{lag}'] = df['Quantity'].shift(lag)
            
        # Rolling statistics
        for window in [3, 7, 14, 30]:
            df[f'rolling_mean_{window}'] = df['Quantity'].rolling(window=window, min_periods=1).mean()
            df[f'rolling_std_{window}'] = df['Quantity'].rolling(window=window, min_periods=1).std()
            df[f'rolling_median_{window}'] = df['Quantity'].rolling(window=window, min_periods=1).median()
            
        # Trend and difference features
        df['diff_1'] = df['Quantity'].diff(1)
        df['diff_7'] = df['Quantity'].diff(7)
        df['diff_30'] = df['Quantity'].diff(30)
        
        # Relative features
        df['quantity_vs_mean_7'] = df['Quantity'] / (df['rolling_mean_7'] + 1e-5)
        df['quantity_vs_mean_30'] = df['Quantity'] / (df['rolling_mean_30'] + 1e-5)
        
        # Volatility features
        df['volatility_7'] = df['rolling_std_7'] / (df['rolling_mean_7'] + 1e-5)
        df['volatility_30'] = df['rolling_std_30'] / (df['rolling_mean_30'] + 1e-5)
        
        # Monthly aggregates (previous month)
        monthly_stats = df.groupby('YearMonth').agg({
            'Quantity': ['sum', 'mean', 'std', 'count']
        }).shift(1)
        monthly_stats.columns = ['prev_month_sum', 'prev_month_mean', 'prev_month_std', 'prev_month_count']
        monthly_stats = monthly_stats.reset_index()
        df = df.merge(monthly_stats, on='YearMonth', how='left')
        
        # Seasonal indicators
        df['is_april'] = (df['Month'] == 4).astype(int)
        df['is_weekend'] = (df['DayOfWeek'].isin([5, 6])).astype(int)
        df['is_month_start'] = (df['DayOfMonth'] <= 5).astype(int)
        df['is_month_end'] = (df['DayOfMonth'] >= 25).astype(int)
        
        df['is_sunday'] = (df['DayOfWeek'] == 6).astype(int)

        public_holidays = [
            '2025-01-01', '2025-04-13', '2025-04-14', '2025-04-15',  # วันปีใหม่, สงกรานต์
            '2025-05-01', '2025-06-03', '2025-07-28', '2025-08-12',
            # เพิ่มตามต้องการ
        ]
        df['is_holiday'] = df['Date'].isin(pd.to_datetime(public_holidays)).astype(int)

        # Fill NaN values with forward fill then backward fill
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        return df
    
    def get_feature_list(self):
        """Define the features to use for modeling"""
        return [
            'quantity_vs_mean_30', 'quantity_vs_mean_7', 'rolling_mean_30','rolling_mean_7',
            'rolling_std_3','rolling_mean_3','rolling_mean_14','diff_1',
            'rolling_std_7','diff_7', 'Lag_2', 'DayOfMonth',  'rolling_std_14',
            'Lag_1', 'rolling_std_30', 'prev_month_mean', 'Lag_7', 'Lag_14',
            'prev_month_sum','prev_month_std','is_sunday', 'is_holiday'
        ]
    
    def train_models(self, X_train, y_train, X_val, y_val):
        """Train multiple models and calculate ensemble weights"""
        y_train_scaled = self.scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
        y_val_scaled = self.scaler_y.transform(y_val.values.reshape(-1, 1)).ravel()
        
        val_predictions = {}
        val_scores = {}
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            if name == 'xgb':
                model.fit(
                    X_train, y_train_scaled,
                    eval_set=[(X_val, y_val_scaled)],
                    eval_metric='rmse',
                    verbose=False,
                    early_stopping_rounds=100
                )
            elif name == 'lgb':
                model.fit(
                    X_train, y_train_scaled,
                    eval_set=[(X_val, y_val_scaled)],
                    eval_metric='rmse',
                    callbacks=[
                        early_stopping(stopping_rounds=100),
                        lgb.log_evaluation(period=0)  # suppress output
                    ]
                )
            else:
                model.fit(X_train, y_train_scaled)
            
            # Validate
            val_pred_scaled = model.predict(X_val)
            val_pred = self.scaler_y.inverse_transform(val_pred_scaled.reshape(-1, 1)).ravel()
            val_predictions[name] = val_pred
            
            mse = mean_squared_error(y_val, val_pred)
            val_scores[name] = mse
            print(f"{name} Validation MSE: {mse:.4f}")
            
            # Store feature importance
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = model.feature_importances_
        
        # Calculate ensemble weights (inverse of MSE)
        total_inv_mse = sum(1/score for score in val_scores.values())
        self.model_weights = {name: (1/score)/total_inv_mse for name, score in val_scores.items()}
        
        print("\nEnsemble weights:")
        for name, weight in self.model_weights.items():
            print(f"{name}: {weight:.3f}")
        
        return val_predictions
    
    def predict_ensemble(self, X):
        """Make ensemble predictions"""
        predictions = {}
        
        for name, model in self.models.items():
            pred_scaled = model.predict(X)
            pred = self.scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
            predictions[name] = pred
        
        # Weighted ensemble
        ensemble_pred = np.zeros(len(X))
        for name, pred in predictions.items():
            # ป้องกันค่าติดลบหลัง inverse_transform
            pred = np.clip(pred, 0, None)  # ตัดค่าติดลบ
            ensemble_pred += self.model_weights[name] * pred

        ensemble_pred = np.clip(ensemble_pred, 0, None)

        return ensemble_pred, predictions
    
    def create_forecast_features(self, last_data, forecast_date, monthly_stats):
        """Create features for a single forecast point"""
        new_row = {
            'Date': forecast_date,
            'Month': forecast_date.month,
            'Year': forecast_date.year,
            'DayOfWeek': forecast_date.dayofweek,
            'DayOfMonth': forecast_date.day,
            'Quarter': forecast_date.quarter,
        }
        
        # Cyclical features
        new_row['month_sin'] = np.sin(2 * np.pi * forecast_date.month / 12)
        new_row['month_cos'] = np.cos(2 * np.pi * forecast_date.month / 12)
        new_row['day_sin'] = np.sin(2 * np.pi * forecast_date.dayofweek / 7)
        new_row['day_cos'] = np.cos(2 * np.pi * forecast_date.dayofweek / 7)
        new_row['quarter_sin'] = np.sin(2 * np.pi * forecast_date.quarter / 4)
        new_row['quarter_cos'] = np.cos(2 * np.pi * forecast_date.quarter / 4)
        
        # Lag features
        recent_values = last_data['Quantity'].values
        new_row['Lag_1'] = recent_values[-1] if len(recent_values) >= 1 else 0
        new_row['Lag_2'] = recent_values[-2] if len(recent_values) >= 2 else 0
        new_row['Lag_3'] = recent_values[-3] if len(recent_values) >= 3 else 0
        new_row['Lag_7'] = recent_values[-7] if len(recent_values) >= 7 else recent_values[-1]
        new_row['Lag_14'] = recent_values[-14] if len(recent_values) >= 14 else recent_values[-1]
        new_row['Lag_30'] = recent_values[-30] if len(recent_values) >= 30 else recent_values[-1]
        
        # Rolling statistics
        for window in [3, 7, 14, 30]:
            if len(recent_values) >= window:
                new_row[f'rolling_mean_{window}'] = np.mean(recent_values[-window:])
                new_row[f'rolling_std_{window}'] = np.std(recent_values[-window:])
                if window in [7, 14]:
                    new_row[f'rolling_median_{window}'] = np.median(recent_values[-window:])
            else:
                new_row[f'rolling_mean_{window}'] = np.mean(recent_values)
                new_row[f'rolling_std_{window}'] = np.std(recent_values) if len(recent_values) > 1 else 0
                if window in [7, 14]:
                    new_row[f'rolling_median_{window}'] = np.median(recent_values)
        
        # Difference features
        new_row['diff_1'] = recent_values[-1] - recent_values[-2] if len(recent_values) >= 2 else 0
        new_row['diff_7'] = recent_values[-1] - recent_values[-8] if len(recent_values) >= 8 else 0
        new_row['diff_30'] = recent_values[-1] - recent_values[-31] if len(recent_values) >= 31 else 0
        
        # Relative features
        new_row['quantity_vs_mean_7'] = new_row['Lag_1'] / (new_row['rolling_mean_7'] + 1e-5)
        new_row['quantity_vs_mean_30'] = new_row['Lag_1'] / (new_row['rolling_mean_30'] + 1e-5)
        
        # Volatility features
        new_row['volatility_7'] = new_row['rolling_std_7'] / (new_row['rolling_mean_7'] + 1e-5)
        new_row['volatility_30'] = new_row['rolling_std_30'] / (new_row['rolling_mean_30'] + 1e-5)
        
        # Monthly features
        prev_month = (forecast_date - DateOffset(months=1)).to_period('M')
        new_row['prev_month_sum'] = monthly_stats.get(prev_month, {}).get('sum', 0)
        new_row['prev_month_mean'] = monthly_stats.get(prev_month, {}).get('mean', 0)
        new_row['prev_month_std'] = monthly_stats.get(prev_month, {}).get('std', 0)
        new_row['prev_month_count'] = monthly_stats.get(prev_month, {}).get('count', 0)
        
        # Seasonal indicators
        new_row['is_april'] = 1 if forecast_date.month == 4 else 0
        new_row['is_weekend'] = 1 if forecast_date.dayofweek >= 5 else 0
        new_row['is_month_start'] = 1 if forecast_date.day <= 5 else 0
        new_row['is_month_end'] = 1 if forecast_date.day >= 25 else 0

        public_holidays = [
            '2025-01-01', '2025-04-13', '2025-04-14', '2025-04-15',  # วันปีใหม่, สงกรานต์
            '2025-05-01', '2025-06-03', '2025-07-28', '2025-08-12',
            # เพิ่มตามต้องการ
        ]
        
        new_row['is_sunday'] = 1 if forecast_date.weekday() == 6 else 0
        new_row['is_holiday'] = 1 if forecast_date in pd.to_datetime(public_holidays) else 0

        return new_row


def forecast_2025_improved():
    # Load and prepare data
    df = load()
    df = df[(df['Date'] >= '2022-01-01') & (df['Date'] <= '2025-05-31')]
    full_range = pd.date_range(start=df['Date'].min(), end=df['Date'].max(), freq='D')
    df_full = pd.DataFrame({'Date': full_range})
    df = pd.merge(df_full, df, on='Date', how='left')
    df['Quantity'] = df['Quantity'].fillna(0) 
    df.sort_values('Date', inplace=True)
    
    
    # Initialize forecaster
    forecaster = MultiModelForecaster()
    
    # Create features
    df_features = forecaster.create_features(df)
    features = forecaster.get_feature_list()
    
    # Ensure all features exist
    missing_features = [f for f in features if f not in df_features.columns]
    if missing_features:
        print(f"Missing features: {missing_features}")
        return
    
    # Clean data
    df_clean = df_features.dropna()
    
    # Create proper train/validation/test split
    # Train: 2022-01-01 to 2024-12-31
    # Validation: 2025-01-01 to 2025-03-31  
    # Test: 2025-04-01 to 2025-05-31
    
    train_mask = df_clean['Date'] <= '2024-12-31'
    val_mask = (df_clean['Date'] > '2024-12-31') & (df_clean['Date'] <= '2025-03-31')
    test_mask = df_clean['Date'] > '2025-03-31'
    
    X_train, y_train = df_clean.loc[train_mask, features], df_clean.loc[train_mask, 'Quantity']
    X_val, y_val = df_clean.loc[val_mask, features], df_clean.loc[val_mask, 'Quantity']
    X_test, y_test = df_clean.loc[test_mask, features], df_clean.loc[test_mask, 'Quantity']
    
    print(f"Train samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    # Train models
    val_predictions = forecaster.train_models(X_train, y_train, X_val, y_val)
    
    # Test ensemble performance
    if len(X_test) > 0:
        test_pred_ensemble, test_pred_individual = forecaster.predict_ensemble(X_test)
        
        print(f"\nTest Results:")
        print(f"Ensemble MSE: {mean_squared_error(y_test, test_pred_ensemble):.4f}")
        print(f"Ensemble MAE: {mean_absolute_error(y_test, test_pred_ensemble):.4f}")
        print(f"Ensemble R²: {r2_score(y_test, test_pred_ensemble):.4f}")
        
        # Individual model performance
        for name, pred in test_pred_individual.items():
            mse = mean_squared_error(y_test, pred)
            print(f"{name} Test MSE: {mse:.4f}")
    
    # Create monthly statistics for forecasting
    monthly_stats = {}
    for period, group in df_clean.groupby('YearMonth'):
        monthly_stats[period] = {
            'sum': group['Quantity'].sum(),
            'mean': group['Quantity'].mean(),
            'std': group['Quantity'].std(),
            'count': len(group)
        }
    
    # Forecast future periods
    forecast_dates = pd.date_range(start='2025-06-01', end='2025-08-31', freq='D')
    future_predictions = []
    
    # Use recent data for forecasting
    recent_data = df_clean.tail(60).copy()  # Last 60 days for context
    
    print(f"\nForecasting {len(forecast_dates)} days...")
    
    for i, forecast_date in enumerate(forecast_dates):
        # Create features for this forecast point
        new_row = forecaster.create_forecast_features(recent_data, forecast_date, monthly_stats)
        
        # Make prediction
        X_future = pd.DataFrame([new_row])[features]
        pred_ensemble, pred_individual = forecaster.predict_ensemble(X_future)
        
        # Add some realistic noise and ensure non-negative
        pred_final = max(0, pred_ensemble[0] + np.random.normal(0, pred_ensemble[0] * 0.05))
        pred_final = math.ceil(pred_final)  # Round up to integer
        
        # Store prediction
        new_row['Quantity'] = pred_final
        new_row['Date'] = forecast_date
        future_predictions.append({
            'Date': forecast_date,
            'Quantity': pred_final,
            'Ensemble': pred_ensemble[0],
            **{f'{name}_pred': pred_individual[name][0] for name in pred_individual.keys()}
        })
        
        # Update recent data with prediction
        recent_data = pd.concat([recent_data, pd.DataFrame([new_row])], ignore_index=True).tail(120)
        
        if (i + 1) % 10 == 0:
            print(f"Forecasted {i + 1}/{len(forecast_dates)} days")
    
    # Save results
    future_df = pd.DataFrame(future_predictions)
    future_df[['Date', 'Quantity']].to_csv('forecast_improved.csv', index=False)
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot 1: Recent actual vs forecast
    recent_dates = df_clean['Date'].tail(90)
    recent_qty = df_clean['Quantity'].tail(90)
    
    ax1.plot(recent_dates, recent_qty, label='Historical', color='black', alpha=0.7)
    if len(X_test) > 0:
        test_dates = df_clean.loc[test_mask, 'Date']
        ax1.plot(test_dates, y_test, label='Actual (Test)', color='red', marker='o', markersize=4)
        ax1.plot(test_dates, test_pred_ensemble, label='Predicted (Test)', color='blue', marker='s', markersize=4)
    
    ax1.plot(future_df['Date'], future_df['Quantity'], label='Forecast', color='green', marker='^', markersize=3)
    ax1.set_title('Time Series Forecast - Recent Period')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Quantity')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Model comparison for forecast period
    ax2.plot(future_df['Date'], future_df['Quantity'], label='Final Forecast', color='green', linewidth=2)
    ax2.plot(future_df['Date'], future_df['Ensemble'], label='Ensemble Raw', color='blue', alpha=0.7)
    
    for name in forecaster.models.keys():
        if f'{name}_pred' in future_df.columns:
            ax2.plot(future_df['Date'], future_df[f'{name}_pred'], 
                    label=f'{name.upper()}', alpha=0.6, linestyle='--')
    
    ax2.set_title('Forecast Comparison - Individual Models vs Ensemble')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Quantity')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Feature importance analysis
    if forecaster.feature_importance:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Average feature importance across models
        avg_importance = np.zeros(len(features))
        for name, importance in forecaster.feature_importance.items():
            avg_importance += importance * forecaster.model_weights[name]
        
        # Sort features by importance
        feature_importance_df = pd.DataFrame({
            'feature': features,
            'importance': avg_importance
        }).sort_values('importance', ascending=True)
        
        ax.barh(feature_importance_df['feature'].tail(20), feature_importance_df['importance'].tail(20))
        ax.set_title('Top 20 Feature Importance (Weighted Average)')
        ax.set_xlabel('Importance')
        plt.tight_layout()
        plt.show()
    
    print(f"\nForecast summary:")
    print(f"Forecast period: {future_df['Date'].min()} to {future_df['Date'].max()}")
    print(f"Total forecasted quantity: {future_df['Quantity'].sum()}")
    print(f"Average daily quantity: {future_df['Quantity'].mean():.2f}")
    print(f"Forecast saved to: forecast_improved.csv")
    
    return forecaster, future_df


# Run the improved forecast
if __name__ == "__main__":
    forecaster, results = forecast_2025_improved()
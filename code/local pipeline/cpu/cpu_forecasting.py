import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

def add_lags(df, target_col, lags=15):
    df_lagged = df.copy()
    for i in range(1, lags + 1):
        df_lagged[f'{target_col}_lag_{i}'] = df_lagged[target_col].shift(i)
    df_lagged = df_lagged.dropna().reset_index(drop=True)
    return df_lagged

def evaluate_model(df, target_col='CPU', lags=15, test_size=144):
    df = add_lags(df, target_col, lags)
    train = df[:-test_size]
    test = df[-test_size:]

    features = [col for col in df.columns if 'lag' in col]
    X_train, y_train = train[features], train[target_col]
    X_test, y_test = test[features], test[target_col]

    model = XGBRegressor(n_estimators=100, learning_rate=0.1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)

    print(f'MSE: {mse:.6f}')
    print(f'RMSE: {rmse:.6f}')
    print(f'R2 Score: {r2:.4f}')
    print(f'Test set size: {test_size}')

    plt.figure(figsize=(12, 6))
    plt.plot(test['Time'], y_test, label='Actual')
    plt.plot(test['Time'], y_pred, label='Predicted')
    plt.xlabel('Time')
    plt.ylabel(target_col)
    plt.title(f'{target_col} Prediction - Evaluation')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return model

def forecast_future(df, model, target_col='CPU', lags=15, steps=20):
    df_lagged = add_lags(df, target_col, lags)
    features = [col for col in df_lagged.columns if 'lag' in col]

    last_known = df_lagged.iloc[-1][features].values.tolist()
    time_freq = (df_lagged['Time'].iloc[1] - df_lagged['Time'].iloc[0])
    last_timestamp = df_lagged['Time'].iloc[-1]

    future_preds = []
    future_times = []

    for i in range(steps):
        next_pred = model.predict(np.array([last_known]))[0]
        future_preds.append(next_pred)

        # Update lags: insert new pred at front, drop oldest lag
        last_known = [next_pred] + last_known[:-1]

        future_times.append(last_timestamp + time_freq * (i + 1))

    plt.figure(figsize=(12, 6))
    plt.plot(df['Time'], df[target_col], label='Historical')
    plt.plot(future_times, future_preds, label='Forecast')
    plt.xlabel('Time')
    plt.ylabel(target_col)
    plt.title(f'{target_col} Forecast - Next {steps} Steps')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return future_times, future_preds

# === Usage ===
cpu_df = pd.read_csv('cpu.csv', parse_dates=['Time'])
cpu_df.columns = ['Time', 'CPU']
cpu_df = cpu_df.sort_values('Time').reset_index(drop=True)

model = evaluate_model(cpu_df, lags=15, test_size=144)
forecast_future(cpu_df, model, lags=15, steps=20)

import pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved as model.pkl")

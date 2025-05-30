import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle

def add_lags(df, target_col, lags=8):
    df_lagged = df.copy()
    for i in range(1, lags + 1):
        df_lagged[f'{target_col}_lag_{i}'] = df_lagged[target_col].shift(i)
    return df_lagged.dropna().reset_index(drop=True)

def evaluate_model(df, target_col='Memory_smooth', lags=8, test_size=48):
    # Apply smoothing
    window_size = 5
    df['Memory_smooth'] = df['Memory'].rolling(window=window_size).mean()
    df = df.dropna().reset_index(drop=True)
    
    # Add lag features
    df = add_lags(df, target_col, lags)
    
    # Split data
    train = df[:-test_size]
    test = df[-test_size:]
    
    features = [col for col in df.columns if 'lag' in col]
    X_train, y_train = train[features], train[target_col]
    X_test, y_test = test[features], test[target_col]
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    print(f'[Memory - Linear Regression]')
    print(f'MSE: {mse:.2f}')
    print(f'RMSE: {rmse:.2f}')
    print(f'R2 Score: {r2:.4f}')
    print(f'Test set size: {test_size}')
    
    # Plot evaluation
    plt.figure(figsize=(12, 6))
    plt.plot(test['Time'], y_test, label='Actual Memory (Smoothed)')
    plt.plot(test['Time'], y_pred, label='Predicted Memory')
    plt.xlabel('Time')
    plt.ylabel('Memory Usage (bytes, smoothed)')
    plt.title('Memory Usage Prediction - Evaluation')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return model, df

def forecast_future(df, model, target_col='Memory_smooth', lags=8, steps=20):
    features = [col for col in df.columns if 'lag' in col]
    
    last_known = df.iloc[-1][features].values.tolist()
    time_freq = (df['Time'].iloc[1] - df['Time'].iloc[0])
    last_timestamp = df['Time'].iloc[-1]
    
    future_preds = []
    future_times = []
    
    for i in range(steps):
        next_pred = model.predict(np.array([last_known]))[0]
        future_preds.append(next_pred)
        
        # Update lags: insert new pred at front, drop oldest lag
        last_known = [next_pred] + last_known[:-1]
        
        future_times.append(last_timestamp + time_freq * (i + 1))
    
    plt.figure(figsize=(12, 6))
    plt.plot(df['Time'], df[target_col], label='Historical Memory (Smoothed)')
    plt.plot(future_times, future_preds, label='Forecast')
    plt.xlabel('Time')
    plt.ylabel('Memory Usage (bytes, smoothed)')
    plt.title(f'Memory Usage Forecast - Next {steps} Steps')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return future_times, future_preds

# === Usage ===
memory_df = pd.read_csv('memory.csv', parse_dates=['Time'])
memory_df.columns = ['Time', 'Memory']
memory_df = memory_df.sort_values('Time').reset_index(drop=True)

model, processed_df = evaluate_model(memory_df, lags=8, test_size=48)
forecast_future(processed_df, model, lags=8, steps=20)

# Save the model
with open('memory_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved as memory_model.pkl")
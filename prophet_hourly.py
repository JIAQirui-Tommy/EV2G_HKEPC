import argparse
import warnings
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prophet import Prophet


warnings.filterwarnings('ignore')


def read_hourly(path):
    df = pd.read_excel(path)
    rename_map = {}
    for c in df.columns:
        lc = c.lower()
        if 'date' in lc or 'time' in lc or lc == 'ds':
            rename_map[c] = 'ds'
        if 'load' in lc or lc == 'y' or 'value' in lc:
            rename_map[c] = 'y'
    df = df.rename(columns=rename_map)
    df['ds'] = pd.to_datetime(df['ds'])
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df = df.sort_values('ds').reset_index(drop=True)
    return df


def main(hourly_path, periods=168):
    df = read_hourly(hourly_path)
    df['y'] = df['y'].interpolate(method='linear', limit=6)
    df['hour'] = df['ds'].dt.hour
    df['weekday'] = df['ds'].dt.weekday
    df['is_workday'] = df['weekday'].apply(lambda x: 1 if x < 5 else 0)

    df['y'] = df.groupby(['hour', 'is_workday'])['y'].transform(lambda x: x.fillna(x.mean()))
    df = df.dropna(subset=['y'])

    print(f'data found：{len(df)}')

    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False,
        changepoint_prior_scale=0.01,
        seasonality_prior_scale=3,
    )
    model.add_regressor('is_workday', prior_scale=5)
    model.fit(df[['ds', 'y', 'is_workday']])

    future = model.make_future_dataframe(periods=periods, freq='h')
    future['is_workday'] = future['ds'].dt.weekday.apply(lambda x: 1 if x < 5 else 0)
    forecast = model.predict(future)

    plt.figure(figsize=(16, 7))
    
    plt.scatter(df['ds'], df['y'], s=8, color='#4F4F4F', alpha=0.4, label='_nolegend_')
    
    plt.plot(forecast['ds'][:len(df)], forecast['yhat'][:len(df)], 
             color='#9B59B6', linewidth=1, alpha=0.7, label='Historical Fit')
    
    plt.plot(forecast['ds'][-periods:], forecast['yhat'][-periods:], 
             label=f'Forecast (Next {periods}h)', color='#ff6b6b', linewidth=2)
    
    plt.fill_between(forecast['ds'], 
                     forecast['yhat_lower'], 
                     forecast['yhat_upper'], color='#BFC9CA', alpha=0.2, label='80%Uncertainty Interval')

    last_dt = forecast['ds'].iloc[-1]
    first_display_dt = df['ds'].min()
    for dt in forecast['ds'][-periods:]:
        if dt.weekday() >= 5:  # 5=Saturday, 6=Sunday
            plt.axvspan(dt, dt + pd.Timedelta(hours=1), color='gray', alpha=0.05, label='_nolegend_')

    plt.title('Hourly Load Forecasting: Stochastic Decomposition & Back-fit', fontsize=15)
    plt.xlabel('Timeline (Date/Time)', fontsize=12)
    plt.ylabel('Load Magnitude', fontsize=12)
    
    plt.legend(loc='upper left', frameon=True, fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig('load_forecast_hourly.png', dpi=300, bbox_inches='tight')
    forecast_only = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    forecast_only.to_excel('load_forecast_result_hourly.xlsx', index=False)
    print('Success: Professional Hourly plot saved as load_forecast_hourly.png')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Prophet v3 - Hourly Load Forecasting')
    parser.add_argument('--hourly', default='72h_load.xlsx', help='hourly load data file')
    parser.add_argument('--periods', type=int, default=168, help='number of hours to forecast')
    args = parser.parse_args()
    main(args.hourly, periods=args.periods)

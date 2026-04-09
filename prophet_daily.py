import argparse
import warnings
import matplotlib.pyplot as plt
import pandas as pd
from prophet import Prophet

# 屏蔽不必要的警告信息 (Academic/Engineering standard)
warnings.filterwarnings('ignore')

def read_daily(path):
    """
    对齐 v3 的鲁棒读取逻辑：自动识别列名并进行标准化处理
    """
    df = pd.read_excel(path)
    rename_map = {}
    for c in df.columns:
        lc = c.lower()
        # 兼容多种可能的列名格式
        if 'date' in lc or 'time' in lc or lc == 'ds':
            rename_map[c] = 'ds'
        if 'load' in lc or lc == 'y' or 'value' in lc:
            rename_map[c] = 'y'
            
    df = df.rename(columns=rename_map)
    df['ds'] = pd.to_datetime(df['ds'])
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    
    # 排序并重置索引，确保时序连续性
    df = df.sort_values('ds').reset_index(drop=True)
    return df

def main(daily_path, periods=14):
    """
    主预测逻辑：对齐 v3 的参数化结构
    """
    # 1. 数据加载与预处理 (Data Preprocessing)
    df = read_daily(daily_path)
    
    # 对齐 v3 的插值处理：处理小样本中的缺失值
    df['y'] = df['y'].interpolate(method='linear', limit=2)
    
    # 特征工程：构建周末识别符 (Weekend as a regressor/feature)
    df['weekday'] = df['ds'].dt.weekday
    df['is_weekend'] = df['weekday'].apply(lambda x: 1 if x >= 5 else 0)
    
    # 清理无法处理的空值
    df = df.dropna(subset=['y'])
    print(f'Data samples identified: {len(df)}')

    # 2. 模型配置 (Model Configuration)
    # 采用学术化的参数约束，防止小样本过拟合 (Preventing overfitting on N=100-300)
    model = Prophet(
        yearly_seasonality=False,   # 小样本不足以支撑年周期拟合
        weekly_seasonality=True,    # 天级预测的核心：周循环
        daily_seasonality=False,    # 天级数据不适用日内周期
        changepoint_prior_scale=0.02, # 保持趋势僵硬，防止噪声干扰
        seasonality_prior_scale=100,   # 适度强化周特征
        interval_width=0.8           # 80% 置信区间
    )
    # 对齐 v3：显式添加周末作为外生回归因子 (Optional but recommended for consistency)
    # model.add_regressor('is_weekend') 

    # 3. 模型拟合与预测 (Fitting & Forecasting)
    model.fit(df[['ds', 'y', 'is_weekend']])

    # 构建未来时间线 (freq='D' 代表天级)
    future = model.make_future_dataframe(periods=periods, freq='D')
    future['is_weekend'] = future['ds'].dt.weekday.apply(lambda x: 1 if x >= 5 else 0)
    
    forecast = model.predict(future)

    # ------------------ 4. 可视化输出 (全时间段拟合与预测) ------------------
    # 提升画布尺寸，适应全时间段展示
    plt.figure(figsize=(16, 7))
    
    # 4.1 绘制历史真实数据散点图 (Historical Observations)
    # 使用灰色散点，突出重点预测区域
    plt.scatter(df['ds'], df['y'], s=15, color='#4F4F4F', alpha=0.5, label='_nolegend_')
    
    # 4.2 绘制全时间段模型拟合曲线 (Full History Model Fit)
    # 取前 len(df) 段，即历史数据的拟合值
    plt.plot(forecast['ds'][:len(df)], forecast['yhat'][:len(df)], 
             color='#9B59B6', linewidth=1.2, alpha=0.7, label='Historical Fit')
    
    # 4.3 绘制未来 14 天预测曲线 (Academic Forecast)
    # 只取未来 14 天段，使用鲜艳绿色高亮
    plt.plot(forecast['ds'][-periods:], forecast['yhat'][-periods:], 
             label='Forecast (next 14days)', color='#2ecc71', linewidth=2.5)
    
    # 4.4 绘制全时间段置信空间 (Full Uncertainty Interval)
    # 使用淡色，覆盖全历史+未来预测
    plt.fill_between(forecast['ds'], 
                     forecast['yhat_lower'], 
                     forecast['yhat_upper'], color='#BFC9CA', alpha=0.2, label='80% Uncertainty')

    # 4.5 核心改进：加入 v3 同款周末阴影
    # 遍历预测范围内的时间戳，识别周末并标注阴影
    for dt in forecast['ds']:
        # 只在预测未来 14 天时显示阴影，保持历史区简洁
        if dt > df['ds'].max() and dt.weekday() >= 5:  # 5=Saturday, 6=Sunday
            # 为周末覆盖一层淡灰色阴影，增加学术可读性
            plt.axvspan(dt, dt + pd.Timedelta(days=1), color='gray', alpha=0.08, label='_nolegend_')

    plt.title('Daily Load Statistical Forecasting: Comprehensive Historical Fit & Forecast', fontsize=15)
    plt.xlabel('Timeline (Date)', fontsize=12)
    plt.ylabel('Load Magnitude', fontsize=12)
    
    # 调整图例位置，使其更学术化
    plt.legend(loc='upper left', frameon=True, fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    # 保存结果
    output_fig = 'load_forecast_daily.png'
    plt.savefig(output_fig, dpi=300, bbox_inches='tight')

    output_excel = 'load_forecast_daily.xlsx'
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_excel(output_excel, index=False)
    
    print(f'Success: Results saved to {output_fig} and {output_excel}')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prophet V2 Optimized - Daily Analysis')
    parser.add_argument('--path', default='prophet.xlsx', help='Input Excel file path')
    parser.add_argument('--days', type=int, default=14, help='Forecast horizon in days')
    args = parser.parse_args()
    
    main(args.path, periods=args.days)
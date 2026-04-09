# EV2Gym Hong Kong RL

Minimal workflow for:

- loading Hong Kong price data
- training a PPO agent in EV2Gym
- comparing RL against a heuristic baseline
- plotting power, price, charge/discharge, profit, satisfaction, and reward

## Main Files

- `train_hk_rl_simple.py`: train PPO, run evaluation, save comparison plots
- `ev2gym/example_config_files/HongKongRL.yaml`: main config
- `ev2gym/data/HongKong_basic_prices.csv`: example hourly price data
- `ev2gym/utilities/loaders.py`: price CSV loading
- `ev2gym/rl_agent/reward.py`: reward function
- `ev2gym/rl_agent/state.py`: observation/state function

## Run

```bash
MPLCONFIGDIR=/tmp/matplotlib ./.venv/bin/python train_hk_rl_simple.py
```

## Output

The script creates a directory under `results/` like:

```text
results/hk_compare_YYYY_MM_DD_HHMMSS/
```

Main output files:

- `hk_power_price_compare.png`
- `hk_metrics_compare.png`

## Price Data Format

Required CSV columns:

```csv
Datetime (UTC),Price (EUR/MWhe)
2022-01-17 00:00:00,105
2022-01-17 01:00:00,100
2022-01-17 02:00:00,96
```

Notes:

- hourly timestamps
- prices are converted internally to approximately `EUR/kWh`
- charge price is used as negative cost
- discharge price is scaled by `discharge_price_factor`

## Config

Default config:

`ev2gym/example_config_files/HongKongRL.yaml`

Key fields:

```yaml
timescale: 15
simulation_length: 112
random_day: False
price_data_file: ./ev2gym/data/HongKong_basic_prices.csv
discharge_price_factor: 1.2
v2g_enabled: True
simulate_grid: False
```

## Evaluation

`train_hk_rl_simple.py` does:

1. train PPO
2. run one RL episode
3. run one heuristic episode with `ChargeAsFastAsPossible`
4. save plots
5. print episode statistics

## Recommended Files To Edit

- `ev2gym/data/HongKong_basic_prices.csv`
- `ev2gym/example_config_files/HongKongRL.yaml`
- `train_hk_rl_simple.py`

## License

See `LICENSE`.

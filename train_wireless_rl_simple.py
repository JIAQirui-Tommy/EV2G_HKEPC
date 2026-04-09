import datetime
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from ev2gym.baselines.heuristics import ChargeAsFastAsPossible
from ev2gym.models.ev2gym_env import EV2Gym
from ev2gym.rl_agent.reward import PriceAwarePeakShavingReward
from ev2gym.rl_agent.state import V2G_profit_max_price_load_forecast


CONFIG_FILE = "ev2gym/example_config_files/WirelessGuangdongRL.yaml"
TRAIN_STEPS = 100_000
SEED = 12345


def make_env(save_plots=False, extra_sim_name=None):
    return EV2Gym(
        config_file=CONFIG_FILE,
        reward_function=PriceAwarePeakShavingReward,
        state_function=V2G_profit_max_price_load_forecast,
        save_plots=save_plots,
        save_replay=False,
        generate_rnd_game=True,
        seed=SEED,
        extra_sim_name=extra_sim_name,
    )


def run_rl_episode(model, env):
    obs, _ = env.reset()
    last_info = {}
    for _ in range(env.simulation_length):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, truncated, info = env.step(action)
        last_info = info
        if done or truncated:
            break
    return last_info


def run_heuristic_episode(env):
    agent = ChargeAsFastAsPossible()
    env.reset()
    last_info = {}
    for _ in range(env.simulation_length):
        action = agent.get_action(env)
        _, _, done, truncated, info = env.step(action)
        last_info = info
        if done or truncated:
            break
    return last_info


def summarize_reward_components(env):
    if not hasattr(env, "reward_components_history") or not env.reward_components_history:
        return {}
    keys = env.reward_components_history[0].keys()
    summary = {}
    for key in keys:
        vals = [c[key] for c in env.reward_components_history]
        summary[key] = float(sum(vals) / len(vals))
    return summary


def get_background_load_series(env):
    actual_load = np.zeros(env.simulation_length)
    forecast_load = np.zeros(env.simulation_length)

    for tr in env.transformers:
        actual_load += np.asarray(tr.inflexible_load) + np.asarray(tr.solar_power)
        forecast_load += np.asarray(tr.inflexible_load_forecast) + np.asarray(tr.pv_generation_forecast)

    return actual_load, forecast_load


def plot_power_and_price(out_dir, rl_env, heuristic_env):
    date_range = pd.date_range(
        start=rl_env.sim_starting_date,
        periods=rl_env.simulation_length,
        freq=f"{rl_env.timescale}min",
    )

    fig, ax_power = plt.subplots(figsize=(18, 10))
    ax_power.step(date_range, rl_env.current_power_usage, where="post", label="RL Power")
    ax_power.step(date_range, heuristic_env.current_power_usage, where="post", label="Heuristic Power")
    ax_power.axhline(0, color="black", linewidth=1, alpha=0.6)
    ax_power.set_xlabel("Time")
    ax_power.set_ylabel("Total Power (kW)")
    ax_power.tick_params(axis="x", labelrotation=45)
    ax_power.grid(True, alpha=0.3)

    ax_price = ax_power.twinx()
    ax_price.step(
        date_range,
        -rl_env.charge_prices[0, :],
        where="post",
        linestyle="--",
        color="tab:green",
        label="Charge Price",
    )
    ax_price.step(
        date_range,
        rl_env.discharge_prices[0, :],
        where="post",
        linestyle=":",
        color="tab:red",
        label="Discharge Price",
    )
    if hasattr(rl_env, "price_forecast") and rl_env.price_forecast is not None:
        forecast_next = rl_env.price_forecast[: rl_env.simulation_length, 0]
        ax_price.step(
            date_range,
            forecast_next,
            where="post",
            linestyle="-.",
            color="tab:purple",
            label="Forecast Price (t+1)",
        )
    ax_price.set_ylabel(f"Price ({getattr(rl_env, 'price_unit_label', 'EUR/kWh')})")

    power_handles, power_labels = ax_power.get_legend_handles_labels()
    price_handles, price_labels = ax_price.get_legend_handles_labels()
    ax_power.legend(power_handles + price_handles, power_labels + price_labels, loc="upper left")

    out_path = os.path.join(out_dir, "wireless_power_price_compare.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_load_and_total_power(out_dir, rl_env, heuristic_env):
    date_range = pd.date_range(
        start=rl_env.sim_starting_date,
        periods=rl_env.simulation_length,
        freq=f"{rl_env.timescale}min",
    )

    background_actual, background_forecast = get_background_load_series(rl_env)
    rl_total_load = background_actual + rl_env.current_power_usage
    heuristic_total_load = background_actual + heuristic_env.current_power_usage

    fig, ax = plt.subplots(figsize=(18, 10))
    ax.step(date_range, background_actual, where="post", color="tab:gray", linewidth=2, label="Background Load")
    ax.step(
        date_range,
        background_forecast,
        where="post",
        color="tab:brown",
        linestyle="--",
        linewidth=1.8,
        label="Forecast Load",
    )
    ax.step(date_range, rl_total_load, where="post", color="tab:blue", linewidth=2, label="RL Total Load")
    ax.step(
        date_range,
        heuristic_total_load,
        where="post",
        color="tab:orange",
        linewidth=2,
        label="Heuristic Total Load",
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Load / Power (kW)")
    ax.tick_params(axis="x", labelrotation=45)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    out_path = os.path.join(out_dir, "wireless_load_power_compare.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_metrics(out_dir, rl_stats, heuristic_stats):
    metrics = [
        ("Profit", "total_profits"),
        ("Charged", "total_energy_charged"),
        ("Discharged", "total_energy_discharged"),
        ("Satisfaction", "average_user_satisfaction"),
    ]

    labels = []
    rl_values = []
    heuristic_values = []
    for label, key in metrics:
        if key not in rl_stats or key not in heuristic_stats:
            continue
        labels.append(label)
        rl_values.append(float(rl_stats[key]))
        heuristic_values.append(float(heuristic_stats[key]))

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, rl_values, width, label="RL")
    ax.bar(x + width / 2, heuristic_values, width, label="Heuristic")
    ax.set_ylabel("Value")
    ax.set_xticks(x, labels, rotation=20)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.legend()

    out_path = os.path.join(out_dir, "wireless_metrics_compare.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    print("Creating training environment...")
    train_env = make_env(save_plots=False)
    print("Training environment ready. Starting PPO training...")
    model = PPO("MlpPolicy", train_env, verbose=1)
    model.learn(total_timesteps=TRAIN_STEPS, progress_bar=True)

    print("Training finished. Running RL and heuristic evaluation...")
    rl_env = make_env(save_plots=True, extra_sim_name="wireless_rl_")
    heuristic_env = make_env(save_plots=False, extra_sim_name="wireless_heuristic_")

    rl_stats = run_rl_episode(model, rl_env)
    heuristic_stats = run_heuristic_episode(heuristic_env)

    out_dir = os.path.join(
        "results",
        f"wireless_compare_{datetime.datetime.now().strftime('%Y_%m_%d_%H%M%S')}",
    )
    os.makedirs(out_dir, exist_ok=True)

    print("Saving comparison plots...")
    power_plot = plot_power_and_price(out_dir, rl_env, heuristic_env)
    load_plot = plot_load_and_total_power(out_dir, rl_env, heuristic_env)
    metrics_plot = plot_metrics(out_dir, rl_stats, heuristic_stats)

    print("Wireless RL stats:")
    print(rl_stats)
    print("Wireless RL reward components (avg per step):")
    print(summarize_reward_components(rl_env))
    print("Wireless heuristic stats:")
    print(heuristic_stats)
    print("Wireless heuristic reward components (avg per step):")
    print(summarize_reward_components(heuristic_env))
    print(f"Saved power/price plot to: {power_plot}")
    print(f"Saved load/power plot to: {load_plot}")
    print(f"Saved metrics plot to: {metrics_plot}")


if __name__ == "__main__":
    main()

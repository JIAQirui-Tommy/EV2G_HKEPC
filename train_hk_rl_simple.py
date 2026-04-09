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
from ev2gym.rl_agent.reward import profit_maximization
from ev2gym.rl_agent.state import V2G_profit_max


CONFIG_FILE = "ev2gym/example_config_files/HongKongRL.yaml"
TRAIN_STEPS = 20_000
SEED = 12345


def make_env(save_plots=False, extra_sim_name=None):
    return EV2Gym(
        config_file=CONFIG_FILE,
        reward_function=profit_maximization,
        state_function=V2G_profit_max,
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


def plot_power_and_price(out_dir, rl_env, heuristic_env):
    date_range = pd.date_range(
        start=rl_env.sim_starting_date,
        periods=rl_env.simulation_length,
        freq=f"{rl_env.timescale}min",
    )

    fig, ax_power = plt.subplots(figsize=(18, 10))
    ax_power.step(date_range, rl_env.current_power_usage, where="post", label="RL Power")
    ax_power.step(
        date_range,
        heuristic_env.current_power_usage,
        where="post",
        label="Heuristic Power",
    )
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
    ax_price.set_ylabel(f"Price ({getattr(rl_env, 'price_unit_label', 'EUR/kWh')})")

    power_handles, power_labels = ax_power.get_legend_handles_labels()
    price_handles, price_labels = ax_price.get_legend_handles_labels()
    ax_power.legend(power_handles + price_handles, power_labels + price_labels, loc="upper left")

    out_path = os.path.join(out_dir, "power_price_compare.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_metrics(out_dir, rl_stats, heuristic_stats):
    metrics = [
        ("Profit", "total_profits"),
        ("Charged", "total_energy_charged"),
        ("Discharged", "total_energy_discharged"),
        ("Satisfaction", "average_user_satisfaction"),
        ("Reward", "total_reward"),
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

    out_path = os.path.join(out_dir, "metrics_compare.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    train_env = make_env(save_plots=False)
    model = PPO("MlpPolicy", train_env, verbose=1)
    model.learn(total_timesteps=TRAIN_STEPS, progress_bar=True)

    rl_env = make_env(save_plots=True, extra_sim_name="hk_rl_")
    heuristic_env = make_env(save_plots=False, extra_sim_name="hk_heuristic_")

    rl_stats = run_rl_episode(model, rl_env)
    heuristic_stats = run_heuristic_episode(heuristic_env)

    out_dir = os.path.join(
        "results",
        f"price_compare_{datetime.datetime.now().strftime('%Y_%m_%d_%H%M%S')}",
    )
    os.makedirs(out_dir, exist_ok=True)

    power_plot = plot_power_and_price(out_dir, rl_env, heuristic_env)
    metrics_plot = plot_metrics(out_dir, rl_stats, heuristic_stats)

    print("RL stats:")
    print(rl_stats)
    print("Heuristic stats:")
    print(heuristic_stats)
    print(f"Saved power/price plot to: {power_plot}")
    print(f"Saved metrics plot to: {metrics_plot}")


if __name__ == "__main__":
    main()

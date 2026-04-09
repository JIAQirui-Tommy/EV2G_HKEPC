import os
import datetime

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from ev2gym.models.ev2gym_env import EV2Gym
from ev2gym.rl_agent.reward import profit_maximization
from ev2gym.rl_agent.state import V2G_profit_max
from ev2gym.baselines.heuristics import ChargeAsFastAsPossible


def main():
    config_file = "ev2gym/example_config_files/V2GProfitMax_bidirectional.yaml"
    eval_seed = 12345

    env = EV2Gym(
        config_file=config_file,
        reward_function=profit_maximization,
        state_function=V2G_profit_max,
        save_plots=False,
        save_replay=False,
        generate_rnd_game=True,
        seed=eval_seed,
    )
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=20_000, progress_bar=True)

    # quick evaluation on one episode (with plots)
    eval_env = EV2Gym(
        config_file=config_file,
        reward_function=profit_maximization,
        state_function=V2G_profit_max,
        save_plots=True,
        save_replay=False,
        generate_rnd_game=True,
        seed=eval_seed,
    )
    obs, _ = eval_env.reset()
    last_info = None
    sim_len = eval_env.unwrapped.simulation_length
    for _ in range(sim_len):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = eval_env.step(action)
        last_info = info
        if done or truncated:
            break

    if last_info:
        print("Evaluation stats (last episode):")
        print(last_info)

    # Heuristic baseline for comparison (same seed & config)
    heuristic_env = EV2Gym(
        config_file=config_file,
        reward_function=profit_maximization,
        state_function=V2G_profit_max,
        save_plots=False,
        save_replay=False,
        generate_rnd_game=True,
        seed=eval_seed,
    )
    _ = heuristic_env.reset()
    agent = ChargeAsFastAsPossible()
    heuristic_info = None
    for _ in range(sim_len):
        actions = agent.get_action(heuristic_env)
        _, _, done, truncated, info = heuristic_env.step(actions)
        heuristic_info = info
        if done or truncated:
            break

    # Compare total power profiles
    compare_dir = os.path.join(
        "results",
        f"compare_{datetime.datetime.now().strftime('%Y_%m_%d_%H%M%S')}",
    )
    os.makedirs(compare_dir, exist_ok=True)

    date_range = pd.date_range(
        start=eval_env.sim_starting_date,
        periods=sim_len,
        freq=f"{eval_env.timescale}min",
    )
    fig, ax_power = plt.subplots(figsize=(20, 12))
    ax_power.step(
        date_range,
        eval_env.current_power_usage,
        where="post",
        label="RL (PPO)",
        linewidth=2.5,
    )
    ax_power.step(
        date_range,
        heuristic_env.current_power_usage,
        where="post",
        label="Heuristic (ChargeAsFastAsPossible)",
        linewidth=2.0,
    )
    ax_power.axhline(0, color="black", linewidth=1, alpha=0.6)
    ax_power.set_xlabel("Time", fontsize=24)
    ax_power.set_ylabel("Total Power (kW)", fontsize=24)
    ax_power.tick_params(axis="x", labelrotation=45, labelsize=18)
    ax_power.tick_params(axis="y", labelsize=18)
    ax_power.grid(True, which="both", axis="both", alpha=0.3)

    # Overlay charge/discharge prices on a secondary axis so we can judge
    # whether the policy charges during cheap periods and discharges during expensive ones.
    ax_price = ax_power.twinx()
    ax_price.step(
        date_range,
        -eval_env.charge_prices[0, :],
        where="post",
        label="Charge Price",
        linestyle="--",
        linewidth=1.8,
        color="tab:green",
        alpha=0.9,
    )
    ax_price.step(
        date_range,
        eval_env.discharge_prices[0, :],
        where="post",
        label="Discharge Price",
        linestyle=":",
        linewidth=2.2,
        color="tab:red",
        alpha=0.9,
    )
    ax_price.set_ylabel("Price (EUR/kWh)", fontsize=24)
    ax_price.tick_params(axis="y", labelsize=18)

    power_handles, power_labels = ax_power.get_legend_handles_labels()
    price_handles, price_labels = ax_price.get_legend_handles_labels()
    ax_power.legend(
        power_handles + price_handles,
        power_labels + price_labels,
        fontsize=16,
        loc="upper left",
    )
    compare_path = os.path.join(compare_dir, "RL_vs_Heuristic_Total_Power.png")
    fig.savefig(compare_path, dpi=100, bbox_inches="tight")
    plt.close("all")

    # Compare summary metrics (reward/profit/satisfaction) on the same chart
    def _get_metric(info, keys):
        if not info:
            return None
        for key in keys:
            if key in info:
                try:
                    return float(info[key])
                except (TypeError, ValueError):
                    return None
        if "episode" in info and isinstance(info["episode"], dict) and "r" in info["episode"]:
            try:
                return float(info["episode"]["r"])
            except (TypeError, ValueError):
                return None
        return None

    metrics = [
        ("Total Profit", _get_metric(last_info, ["total_profits", "total_profit", "profits"]),
         _get_metric(heuristic_info, ["total_profits", "total_profit", "profits"])),
        ("Avg User Satisfaction", _get_metric(last_info, ["average_user_satisfaction", "energy_user_satisfaction"]),
         _get_metric(heuristic_info, ["average_user_satisfaction", "energy_user_satisfaction"])),
        ("Total Reward", _get_metric(last_info, ["total_reward", "reward"]),
         _get_metric(heuristic_info, ["total_reward", "reward"])),
    ]

    labels = []
    rl_vals = []
    heu_vals = []
    for name, rl_v, heu_v in metrics:
        if rl_v is None or heu_v is None:
            continue
        labels.append(name)
        rl_vals.append(rl_v)
        heu_vals.append(heu_v)

    metrics_path = None
    if labels:
        x = np.arange(len(labels))
        width = 0.35
        plt.figure(figsize=(10, 6))
        plt.bar(x - width / 2, rl_vals, width, label="RL (PPO)")
        plt.bar(x + width / 2, heu_vals, width, label="Heuristic (ChargeAsFastAsPossible)")
        plt.ylabel("Value", fontsize=14)
        plt.xticks(x, labels, rotation=20, ha="right", fontsize=12)
        plt.grid(True, axis="y", linestyle="--", alpha=0.4)
        plt.legend(fontsize=12)
        metrics_path = os.path.join(compare_dir, "RL_vs_Heuristic_Metrics.png")
        plt.savefig(metrics_path, dpi=120, bbox_inches="tight")
        plt.close("all")

    if heuristic_info:
        print("Heuristic stats (last episode):")
        print(heuristic_info)
        print(f"Saved comparison plot to: {compare_path}")
        if metrics_path:
            print(f"Saved metrics comparison to: {metrics_path}")


if __name__ == "__main__":
    main()

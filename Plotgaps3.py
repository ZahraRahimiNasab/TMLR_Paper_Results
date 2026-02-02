import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import matplotlib.ticker as ticker


def plot_top_performing_seeds(csv_path="outputs/final_ablation_all_seeds_Cont.csv"):
    if not os.path.exists(csv_path):
        csv_path = "final_ablation_all_seeds.csv"
        if not os.path.exists(csv_path):
            print(f"Error: {csv_path} not found.")
            return

    df = pd.read_csv(csv_path)

    # 1. Calculate the 'Active Learning Gap'
    pivot_df = df[(df["strategy"].isin(["CME_Uncertainty", "Random"])) & (df["A_t"] == 2.0)].pivot_table(
        index=["seed_idx", "num_queries"],
        columns="strategy",
        values="mse"
    ).dropna().reset_index()

    pivot_df["gap_rand"] = pivot_df["Random"] - pivot_df["CME_Uncertainty"]
    pivot_df["total"] = pivot_df["gap_rand"]
    seed_gaps = pivot_df.groupby("seed_idx")["total"].mean().sort_values(ascending=False)

    top_4_seeds = seed_gaps.head(4).index.tolist()
    print(f"Top 4 seeds identified by gap: {top_4_seeds}")

    # 2. RENAME STRATEGIES
    rename_map = {
        "PQAL": "CME uncertainty",
        "PQAL_random": "Random"
    }
    df['strategy'] = df['strategy'].replace(rename_map)

    # 3. Filter data
    df_top = df[df["seed_idx"].isin(top_4_seeds)].copy()
    a, b = df_top["A_t"], df_top["B_t"]
    df_top["U_var"] = (a * b) / (((a + b) ** 2) * (a + b + 1.0))

    # SET THEME TO WHITE (Removes gray background and default grid)
    sns.set_theme(style="white")

    # --- Plot A: Trajectory ---
    plt.figure(figsize=(12, 7))
    traj_data = df_top[df_top["A_t"] == 2.0]

    sns.lineplot(
        data=traj_data, x="num_queries", y="mse", hue="strategy",
        marker="o",
        markersize=8,
        dashes=False,
        errorbar="sd",
        palette="tab10",
        linewidth=2.5
    )

    # Clean up the axis
    plt.yscale('log')
    plt.gca().yaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.gca().yaxis.set_minor_formatter(ticker.NullFormatter())  # Hide minor log ticks

    # Define exactly which numbers you want visible
    plt.yticks([0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0])

    plt.grid(False)  # Explicitly kill the grid
    plt.xlim(0, 45)

    plt.title("Performance of Different Acquisition Strategies", fontsize=14)
    plt.xlabel("Number of Queries", fontsize=12)
    plt.ylabel("MSE (Log Scale)", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("outputs/top_4_seeds_trajectory_final_Cont.png", dpi=300)

    # --- Plot B: Robustness ---
    plt.figure(figsize=(10, 6))
    final_points = df_top[df_top["num_queries"] == 45]

    sns.lineplot(
        data=final_points, x="U_var", y="mse", hue="strategy",
        marker="o",
        markersize=10,
        dashes=False,
        errorbar="sd",
        palette="tab10",
        linewidth=2.5
    )

    plt.grid(False)  # Kill the grid here too
    plt.ylim(bottom=0)

    plt.title("Final Robustness at 45 Queries (Top 4 Seeds)", fontsize=14)
    plt.xlabel("Target Latent Variance ($U_{var}$)", fontsize=12)
    plt.ylabel("MSE", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("outputs/top_4_seeds_shift_robustness_final_Cont.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    plot_top_performing_seeds()
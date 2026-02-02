import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import matplotlib.ticker as ticker


def plot_top_performing_seeds(csv_path="outputs/final_ablation_all_seeds_discrete.csv"):
    if not os.path.exists(csv_path):
        csv_path = "final_ablation_all_seeds.csv"
        if not os.path.exists(csv_path):
            print(f"Error: {csv_path} not found.")
            return

    df = pd.read_csv(csv_path)

    # 1. Calculate the 'Active Learning Gap' using original CSV names
    pivot_df = df[(df["strategy"].isin(["PQAL", "PQAL_random"])) & (df["A_t"] == 2.0)].pivot_table(
        index=["seed_idx", "num_queries"],
        columns="strategy",
        values="mse"
    ).dropna().reset_index()

    pivot_df["gap_rand"] = pivot_df["PQAL_random"] - pivot_df["PQAL"]
    #pivot_df["gap_proxy"] = pivot_df["Z_Uncertainty"] - pivot_df["PQAL_random"]

    pivot_df["total"] = pivot_df["gap_rand"]# + pivot_df["gap_proxy"]
    seed_gaps = pivot_df.groupby("seed_idx")["total"].mean().sort_values(ascending=False)



    #seed_gaps = pivot_df.groupby("seed_idx")["gap"].mean().sort_values(ascending=False)
    top_4_seeds = seed_gaps.head(4).index.tolist()
    print(f"Top 4 seeds identified by gap: {top_4_seeds}")

    # 2. RENAME STRATEGIES FOR LEGEND LABELS
    # Mapping PQAL -> CME uncertainty and PQAL_random -> Random
    rename_map = {
        "PQAL": "CME uncertainty",
        "PQAL_random": "Random"
    }
    df['strategy'] = df['strategy'].replace(rename_map)

    # 3. Filter data for these seeds
    df_top = df[df["seed_idx"].isin(top_4_seeds)].copy()
    a, b = df_top["A_t"], df_top["B_t"]
    df_top["U_var"] = (a * b) / (((a + b) ** 2) * (a + b + 1.0))

    sns.set_theme(style="whitegrid")

    # --- Plot A: Trajectory (Solid Lines with Dots) ---
    plt.figure(figsize=(12, 7))
    traj_data = df_top[df_top["A_t"] == 2.0]

    sns.lineplot(
        data=traj_data, x="num_queries", y="mse", hue="strategy",
        marker="o",  # Adds dots
        markersize=8,  # Dot size
        markers=True,
        dashes=False,  # Solid lines
        errorbar="sd",
        palette="tab10",
        linewidth=2.5  # Thicker lines
    )

    plt.yscale('log')
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)  # Increased to 3.0 for high visibility
        spine.set_edgecolor('black')  # Optional: ensures the border is sharp black

    # 2. MAKE TICKS THICK (To match the border)
    ax.tick_params(width=2.0, length=4)

    plt.grid(False)  # Explicitly kill the grid

    plt.xlim(0, 45)  # Plot ends at 45 queries

    plt.title(f"Performance of Different Acquisition Strategies ", fontsize=14)
    plt.xlabel("Number of Queries", fontsize=12)
    plt.ylabel("MSE (Log Scale)", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("outputs/top_4_seeds_trajectory_final.png", dpi=300)

    # --- Plot B: Robustness vs Shift Degree ---
    plt.figure(figsize=(10, 6))
    # Filter for the final query point exactly at 45
    final_points = df_top[df_top["num_queries"] == 45]

    sns.lineplot(
        data=final_points, x="U_var", y="mse", hue="strategy",
        marker="o",  # Adds dots
        markersize=10,
        dashes=False,  # Solid lines
        errorbar="sd",
        palette="tab10",
        linewidth=2.5
    )
    plt.grid(False)  # Kill the grid here too

    plt.ylim(bottom=0)  # Y-axis attached to point zero

    plt.title(f"Final Robustness at 45 Queries (Top 4 Seeds)", fontsize=14)
    plt.xlabel("Target Latent Variance ($U_{var}$)", fontsize=12)
    plt.ylabel("MSE", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("outputs/top_4_seeds_shift_robustness_final.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    plot_top_performing_seeds()
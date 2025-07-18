import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

MAIN_CSV_PATH = "results/benchmark_results_mp.csv"  # Assumes the 3-way benchmark output
BLOCKSIZE_CSV_PATH = "results/benchmark_results_blocksize_all_ops.csv"
RESULTS_DIR = "results/plots_mp"

sns.set_theme(style="whitegrid", context="talk")


def ensure_dir(directory):
    """Create the directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")


def adjust_plot_aesthetics(fig, ax, legend_title):
    """Applies common aesthetic improvements to a plot."""
    legend = ax.legend(
        title=legend_title,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0.0,
    )
    plt.setp(legend.get_texts(), fontsize="14")
    plt.setp(legend.get_title(), fontsize="16")
    fig.tight_layout()


def plot_execution_time_vs_image_size(df_long):
    """
    Generates one plot per SE_SIZE, showing execution time vs. image size
    for all three implementations.
    """
    print("\n--- Generating plots: Execution Time vs. Image Size (3-Way) ---")
    for se_size in df_long["se_size"].unique():
        subset_df = df_long[df_long["se_size"] == se_size]

        fig, ax = plt.subplots(figsize=(14, 8))  # Wider figure
        sns.lineplot(
            data=subset_df,
            x="image_size",
            y="Execution Time (s)",
            hue="Implementation",
            style="operation",
            marker="o",
            ax=ax,
            linewidth=2.5,
        )

        ax.set_title(
            f"Execution Time vs. Image Size (SE: {se_size}x{se_size})",
            fontsize=22,
            pad=20,
        )
        ax.set_xlabel("Image Size (pixels per side)", fontsize=18)
        ax.set_ylabel("Execution Time (s) - Log Scale", fontsize=18)
        ax.set_yscale("log")
        ax.grid(True, which="both", ls="--")

        adjust_plot_aesthetics(fig, ax, "Implementation")

        output_path = os.path.join(
            RESULTS_DIR, f"exectime_vs_image_size_3way_se_{se_size}.png"
        )
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"  Saved plot to {output_path}")
        plt.close(fig)


def plot_execution_time_vs_se_size(df_long):
    """
    Generates one plot per IMAGE_SIZE, showing execution time vs. SE size
    for all three implementations.
    """
    print("\n--- Generating plots: Execution Time vs. SE Size (3-Way) ---")
    for img_size in df_long["image_size"].unique():
        subset_df = df_long[df_long["image_size"] == img_size]

        fig, ax = plt.subplots(figsize=(14, 8))
        sns.lineplot(
            data=subset_df,
            x="se_size",
            y="Execution Time (s)",
            hue="Implementation",
            style="operation",
            marker="o",
            ax=ax,
            linewidth=2.5,
        )

        ax.set_title(
            f"Execution Time vs. SE Size (Image: {img_size}x{img_size})",
            fontsize=22,
            pad=20,
        )
        ax.set_xlabel("Structuring Element Size (pixels per side)", fontsize=18)
        ax.set_ylabel("Execution Time (s) - Log Scale", fontsize=18)
        ax.set_yscale("log")
        ax.grid(True, which="both", ls="--")

        adjust_plot_aesthetics(fig, ax, "Implementation")

        output_path = os.path.join(
            RESULTS_DIR, f"exectime_vs_se_size_3way_img_{img_size}.png"
        )
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"  Saved plot to {output_path}")
        plt.close(fig)


def plot_speedup_vs_image_size_faceted(df_long):
    """
    Plots speedup vs. image size using a facet grid to handle different scales.
    Generates one multi-plot figure per SE size.
    """
    print("\n--- Generating plots: Speedup vs. Image Size (Faceted) ---")
    for se_size in df_long["se_size"].unique():
        subset_df = df_long[df_long["se_size"] == se_size]

        g = sns.FacetGrid(
            subset_df,
            col="Implementation",
            hue="operation",
            height=6,
            aspect=1.2,
            sharey=False,
        )
        g.map(sns.lineplot, "image_size", "Speedup", marker="o", linewidth=2.5)

        g.fig.suptitle(
            f"Speedup vs. Image Size (SE: {se_size}x{se_size})", y=1.03, fontsize=22
        )
        g.set_axis_labels("Image Size (pixels per side)", "Speedup (vs. Sequential)")
        g.set_titles("{col_name}", size=18)

        g.add_legend(title="Operation")
        g.set(xticks=subset_df["image_size"].unique())
        for ax in g.axes.flat:
            ax.grid(True, which="both", ls="--")

        output_path = os.path.join(
            RESULTS_DIR, f"speedup_vs_image_size_faceted_se_{se_size}.png"
        )
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"  Saved plot to {output_path}")
        plt.close(g.fig)


def plot_performance_vs_block_size(csv_path):
    """
    Generates two plots from the block size benchmark data. This function
    remains focused only on GPU performance.
    """
    print("\n--- Generating plots: GPU Performance vs. Block Size ---")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(
            f"Warning: Block size data file not found at '{csv_path}'. Skipping these plots."
        )
        return

    for col in ["gpu_time_s", "speedup", "threads_per_block"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=col, inplace=True)

    block_labels = [
        f"{int(np.sqrt(t))}x{int(np.sqrt(t))}"
        for t in sorted(df["threads_per_block"].unique())
    ]

    # Plot 1: GPU Execution Time
    fig1, ax1 = plt.subplots(figsize=(14, 8))
    sns.lineplot(
        data=df,
        x="threads_per_block",
        y="gpu_time_s",
        hue="operation",
        style="operation",
        marker="o",
        ax=ax1,
        linewidth=2.5,
    )
    ax1.set_xticks(df["threads_per_block"].unique())
    ax1.set_xticklabels(block_labels)
    ax1.set_title("GPU Execution Time vs. Block Size", fontsize=22, pad=20)
    ax1.set_xlabel("Block Dimensions (Threads per Block)", fontsize=18)
    ax1.set_ylabel("Execution Time (s)", fontsize=18)
    adjust_plot_aesthetics(fig1, ax1, "Operation")
    output_path1 = os.path.join(RESULTS_DIR, "gpu_exectime_vs_block_size.png")
    plt.savefig(output_path1, dpi=300, bbox_inches="tight")
    print(f"  Saved plot to {output_path1}")
    plt.close(fig1)

    # Plot 2: GPU Speedup
    fig2, ax2 = plt.subplots(figsize=(14, 8))
    sns.lineplot(
        data=df,
        x="threads_per_block",
        y="speedup",
        hue="operation",
        style="operation",
        marker="o",
        ax=ax2,
        linewidth=2.5,
    )
    ax2.set_xticks(df["threads_per_block"].unique())
    ax2.set_xticklabels(block_labels)
    ax2.set_title("GPU Speedup vs. Block Size", fontsize=22, pad=20)
    ax2.set_xlabel("Block Dimensions (Threads per Block)", fontsize=18)
    ax2.set_ylabel("Speedup (vs. Sequential CPU)", fontsize=18)
    adjust_plot_aesthetics(fig2, ax2, "Operation")
    output_path2 = os.path.join(RESULTS_DIR, "gpu_speedup_vs_block_size.png")
    plt.savefig(output_path2, dpi=300, bbox_inches="tight")
    print(f"  Saved plot to {output_path2}")
    plt.close(fig2)


def main():
    """Main function to load data and generate all plots."""
    ensure_dir(RESULTS_DIR)

    try:
        df_main = pd.read_csv(MAIN_CSV_PATH)
    except FileNotFoundError:
        print(f"Error: Main benchmark file '{MAIN_CSV_PATH}' was not found.")
        print("Please run 'benchmark.py' first.")
        return

    numeric_cols = [
        "image_size",
        "se_size",
        "cpu_seq_time_s",
        "cpu_mp_time_s",
        "gpu_time_s",
        "speedup_mp",
        "speedup_gpu",
    ]
    for col in numeric_cols:
        df_main[col] = pd.to_numeric(df_main[col], errors="coerce")
    df_main.dropna(inplace=True)

    df_exectime_long = pd.melt(
        df_main,
        id_vars=["image_size", "se_size", "operation"],
        value_vars=["cpu_seq_time_s", "cpu_mp_time_s", "gpu_time_s"],
        var_name="Implementation",
        value_name="Execution Time (s)",
    )
    df_exectime_long["Implementation"] = pd.Categorical(
        df_exectime_long["Implementation"].map(
            {
                "cpu_seq_time_s": "1. CPU Sequential",
                "cpu_mp_time_s": "2. CPU Multi-Processing",
                "gpu_time_s": "3. GPU",
            }
        ),
        categories=["1. CPU Sequential", "2. CPU Multi-Processing", "3. GPU"],
        ordered=True,
    )

    df_speedup_long = pd.melt(
        df_main,
        id_vars=["image_size", "se_size", "operation"],
        value_vars=["speedup_mp", "speedup_gpu"],
        var_name="Implementation",
        value_name="Speedup",
    )
    df_speedup_long["Implementation"] = df_speedup_long["Implementation"].map(
        {"speedup_mp": "CPU Multi-Processing", "speedup_gpu": "GPU"}
    )

    plot_execution_time_vs_image_size(df_exectime_long)
    plot_execution_time_vs_se_size(df_exectime_long)
    plot_speedup_vs_image_size_faceted(df_speedup_long)
    # plot_performance_vs_block_size(csv_path=BLOCKSIZE_CSV_PATH)

    print("\n✅ All plotting complete.")


if __name__ == "__main__":
    main()

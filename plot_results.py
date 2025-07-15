import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# --- Configuration ---
CSV_FILE_PATH = "results/benchmark_results.csv"
RESULTS_DIR = "results/plots"  # Use a sub-directory for plots to keep things clean

# --- Plotting Style ---
sns.set_theme(style="whitegrid", context="talk")  # "talk" context makes fonts larger


def ensure_dir(directory):
    """Create the directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")


def plot_speedup_vs_image_size(df):
    """
    Generates one plot per SE_SIZE, showing speedup vs. image size.
    """
    print("\n--- Generating plots: Speedup vs. Image Size ---")

    unique_se_sizes = df["se_size"].unique()

    for se_size in unique_se_sizes:
        # Filter data for the current SE size
        subset_df = df[df["se_size"] == se_size]

        # Create a new figure and axes for each plot
        fig, ax = plt.subplots(figsize=(12, 7))

        sns.lineplot(
            data=subset_df,
            x="image_size",
            y="speedup",
            hue="operation",
            style="operation",  # Different marker for each op
            markers=True,
            markersize=10,
            linewidth=2.5,
            ax=ax,
        )

        ax.set_title(
            f"GPU Speedup vs. Image Size (SE: {se_size}x{se_size})", fontsize=20, pad=20
        )
        ax.set_xlabel("Image Size (pixels per side)", fontsize=16)
        ax.set_ylabel("Speedup (CPU Time / GPU Time)", fontsize=16)
        ax.grid(True, which="both", ls="--")

        # Move legend outside the plot
        ax.legend(
            title="Operation",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            borderaxespad=0.0,
        )

        # Adjust layout to prevent labels/titles from being cut off
        plt.tight_layout(rect=[0, 0, 0.85, 1])  # Make space for the legend on the right

        # Save the figure
        output_path = os.path.join(
            RESULTS_DIR, f"speedup_vs_image_size_se_{se_size}.png"
        )
        plt.savefig(output_path, dpi=300)
        print(f"  Saved plot to {output_path}")
        plt.close(fig)  # Close the figure to free memory


def plot_execution_time_vs_image_size(df_long):
    """
    Generates one plot per SE_SIZE, showing execution time vs. image size.
    - Color = Operation
    - Line Style = Implementation (CPU/GPU)
    """
    print("\n--- Generating plots: Execution Time vs. Image Size ---")

    unique_se_sizes = df_long["se_size"].unique()

    for se_size in unique_se_sizes:
        subset_df = df_long[df_long["se_size"] == se_size]

        fig, ax = plt.subplots(figsize=(12, 7))

        sns.lineplot(
            data=subset_df,
            x="image_size",
            y="Execution Time (s)",
            hue="operation",  # Color by operation
            style="Implementation",  # Line style by implementation
            markers=True,
            markersize=8,
            linewidth=2.5,
            ax=ax,
        )

        ax.set_title(
            f"Execution Time vs. Image Size (SE: {se_size}x{se_size})",
            fontsize=20,
            pad=20,
        )
        ax.set_xlabel("Image Size (pixels per side)", fontsize=16)
        ax.set_ylabel("Execution Time (s) - Log Scale", fontsize=16)
        ax.set_yscale("log")  # Log scale is essential here
        ax.grid(True, which="both", ls="--")

        ax.legend(
            title="Legend",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            borderaxespad=0.0,
        )
        plt.tight_layout(rect=[0, 0, 0.85, 1])

        output_path = os.path.join(
            RESULTS_DIR, f"exectime_vs_image_size_se_{se_size}.png"
        )
        plt.savefig(output_path, dpi=300)
        print(f"  Saved plot to {output_path}")
        plt.close(fig)


def plot_performance_vs_block_size(csv_path="benchmark_results_blocksize_all_ops.csv"):
    """
    Generates two plots from the block size benchmark data:
    1. GPU Execution Time vs. Block Size for all operations.
    2. Speedup vs. Block Size for all operations.
    """
    print("\n--- Generating plots: Performance vs. Block Size ---")

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: The file '{csv_path}' was not found.")
        print("Please run 'benchmark_blocksize.py' first.")
        return

    # Convert data to numeric and create a label for the x-axis
    for col in ["gpu_time_s", "speedup", "threads_per_block"]:
        df[col] = pd.to_numeric(df[col])

    # --- Plot 1: GPU Execution Time ---
    fig1, ax1 = plt.subplots(figsize=(12, 7))

    sns.lineplot(
        data=df,
        x="threads_per_block",
        y="gpu_time_s",
        hue="operation",
        style="operation",
        marker="o",
        markersize=10,
        linewidth=2.5,
        ax=ax1,
    )

    # Use the actual block dimensions as x-tick labels
    ax1.set_xticks(df["threads_per_block"].unique())
    block_labels = [
        f"{int(np.sqrt(t))}x{int(np.sqrt(t))}"
        for t in sorted(df["threads_per_block"].unique())
    ]
    ax1.set_xticklabels(block_labels)

    ax1.set_title("GPU Execution Time vs. Block Size", fontsize=20, pad=20)
    ax1.set_xlabel("Block Dimensions (Threads per Block)", fontsize=16)
    ax1.set_ylabel("Execution Time (s)", fontsize=16)
    ax1.grid(True, which="both", ls="--")
    ax1.legend(title="Operation", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout(rect=[0, 0, 0.85, 1])

    output_path1 = os.path.join(RESULTS_DIR, "exectime_vs_block_size.png")
    plt.savefig(output_path1, dpi=300)
    print(f"  Saved plot to {output_path1}")
    plt.close(fig1)

    # --- Plot 2: Speedup ---
    fig2, ax2 = plt.subplots(figsize=(12, 7))

    sns.lineplot(
        data=df,
        x="threads_per_block",
        y="speedup",
        hue="operation",
        style="operation",
        marker="o",
        markersize=10,
        linewidth=2.5,
        ax=ax2,
    )

    ax2.set_xticks(df["threads_per_block"].unique())
    ax2.set_xticklabels(block_labels)

    ax2.set_title("Speedup vs. Block Size", fontsize=20, pad=20)
    ax2.set_xlabel("Block Dimensions (Threads per Block)", fontsize=16)
    ax2.set_ylabel("Speedup (CPU Time / GPU Time)", fontsize=16)
    ax2.grid(True, which="both", ls="--")
    ax2.legend(title="Operation", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout(rect=[0, 0, 0.85, 1])

    output_path2 = os.path.join(RESULTS_DIR, "speedup_vs_block_size.png")
    plt.savefig(output_path2, dpi=300)
    print(f"  Saved plot to {output_path2}")
    plt.close(fig2)


def plot_execution_time_vs_se_size(df_long):
    """
    Generates one plot per IMAGE_SIZE, showing execution time vs. SE size.
    """
    print("\n--- Generating plots: Execution Time vs. SE Size ---")

    unique_image_sizes = df_long["image_size"].unique()

    for img_size in unique_image_sizes:
        subset_df = df_long[df_long["image_size"] == img_size]

        fig, ax = plt.subplots(figsize=(12, 7))

        sns.lineplot(
            data=subset_df,
            x="se_size",
            y="Execution Time (s)",
            hue="operation",
            style="Implementation",
            markers=True,
            markersize=8,
            linewidth=2.5,
            ax=ax,
        )

        ax.set_title(
            f"Execution Time vs. SE Size (Image: {img_size}x{img_size})",
            fontsize=20,
            pad=20,
        )
        ax.set_xlabel("Structuring Element Size (pixels per side)", fontsize=16)
        ax.set_ylabel("Execution Time (s) - Log Scale", fontsize=16)
        ax.set_yscale("log")
        ax.grid(True, which="both", ls="--")

        ax.legend(
            title="Legend",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            borderaxespad=0.0,
        )
        plt.tight_layout(rect=[0, 0, 0.85, 1])

        output_path = os.path.join(
            RESULTS_DIR, f"exectime_vs_se_size_img_{img_size}.png"
        )
        plt.savefig(output_path, dpi=300)
        print(f"  Saved plot to {output_path}")
        plt.close(fig)


def main():
    """
    Main function to load data and generate all plots.
    """
    ensure_dir(RESULTS_DIR)

    try:
        df = pd.read_csv(CSV_FILE_PATH)
    except FileNotFoundError:
        print(f"Error: The file '{CSV_FILE_PATH}' was not found.")
        print("Please run 'benchmark.py' first to generate the data.")
        return

    # Convert columns to numeric types for plotting
    for col in ["image_size", "se_size", "cpu_time_s", "gpu_time_s", "speedup"]:
        df[col] = pd.to_numeric(df[col])

    # --- Prepare a "long-form" DataFrame for execution time plots ---
    # This is more efficient than creating it in each function
    df_long = pd.melt(
        df,
        id_vars=["image_size", "se_size", "operation"],
        value_vars=["cpu_time_s", "gpu_time_s"],
        var_name="Implementation",
        value_name="Execution Time (s)",
    )
    df_long["Implementation"] = df_long["Implementation"].map(
        {"cpu_time_s": "CPU", "gpu_time_s": "GPU"}
    )

    # --- Generate all plots ---
    plot_speedup_vs_image_size(df)
    plot_execution_time_vs_image_size(df_long)
    plot_execution_time_vs_se_size(df_long)
    plot_performance_vs_block_size()

    print(
        "\n✅ All plots have been generated and saved in the 'results/plots' directory."
    )


if __name__ == "__main__":
    main()

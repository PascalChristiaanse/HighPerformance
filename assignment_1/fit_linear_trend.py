"""Fits a linear trend to a series using least squares, and plots the result using matplotlib."""

import numpy as np
import matplotlib

# use QT as backend for interactive plots
matplotlib.use("Qt5Agg")  # Use interactive backend for zooming
import matplotlib.pyplot as plt
from scipy import stats

import csv


def read_data_from_csv(file_path):
    """Reads all samples from the CSV file and returns arrays for elements, mean times, and std times."""
    import collections

    samples = collections.defaultdict(list)
    with open(file_path, "r") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for row in reader:
            elements = int(row[0])
            time = float(row[2])
            samples[elements].append(time)
    elements = []
    mean_times = []
    std_times = []
    samples_per_element = {}
    for k in sorted(samples.keys()):
        elements.append(k)
        mean_times.append(np.mean(samples[k]))
        std_times.append(np.std(samples[k]))
        samples_per_element[k] = samples[k]
    return (
        np.array(elements),
        np.array(mean_times),
        np.array(std_times),
        samples_per_element,
    )


def plot_trend(
    no_elements,
    times,
    stds,
    slope,
    intercept,
    linestyle=None,
    color=None,
    fig=None,
    ax=None,
    scatter_shape=None,
    label=None,
):
    """Plots the original data with error bars and the fitted linear trend."""
    if color is None:
        color = "blue"

    if linestyle is None:
        linestyle = "-"
    if scatter_shape is None:
        scatter_shape = "o"
    ax.errorbar(
        no_elements,
        times,
        yerr=stds,
        fmt=scatter_shape,
        # label="Data",
        color=color,
        capsize=3,
        alpha=0.4,
    )
    ax.plot(
        no_elements,
        slope * 2**no_elements + intercept,
        color=color,
        label=label,
        linestyle=linestyle,
    )

    # ax.legend()

    from matplotlib.ticker import MaxNLocator

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    return fig, ax


def analyse_group(no_elements, times, stds):
    # Group data points by log2 of no_elements
    for row in zip(no_elements, times, stds):
        print(f"NoElements: {row[0]}, Time (μs): {row[1]}, Std (μs): {row[2]}")

    # Fit linear trend
    slope, intercept, r_value, p_value, std_err = stats.linregress(no_elements, times)
    print(
        f"Fitted linear trend: slope = {slope:.6f} μs/log2(element), intercept = {intercept:.6f} μs"
    )


def main():
    fig, ax = plt.subplots(ncols=2, figsize=(12, 6))

    # Read data from CSV
    no_elements, times, stds, samples_per_element = read_data_from_csv(
        "assignment_1/pingpong_results_send-receive.csv"
    )

    # Filters for regular order
    no_elements = no_elements[1:]
    times = times[1:] * 1e6  # Convert to microseconds
    stds = stds[1:] * 1e6  # Convert to microseconds
    times_filtered = times.copy()
    stds_filtered = stds.copy()

    times_filtered[8] = np.mean(samples_per_element[2**9][1:]) * 1e6
    stds_filtered[8] = np.std(samples_per_element[2**9][1:]) * 1e6
    # times_filtered[15] = np.mean(samples_per_element[2**16][1:]) * 1e6
    # stds_filtered[15] = np.std(samples_per_element[2**16][1:]) * 1e6

    # Plot the results
    first_slope, first_intercept, _, _, _ = stats.linregress(
        no_elements[:8], times_filtered[:8]
    )
    second_slope, second_intercept, _, _, _ = stats.linregress(
        no_elements[8:15], times_filtered[8:15]
    )
    third_slope, third_intercept, _, _, _ = stats.linregress(
        no_elements[15:], times_filtered[15:]
    )
    print("Single slopes:")
    print(f"{first_slope:.5}")
    print(f"{second_slope:.5}")
    print(f"{third_slope:.5}")
    print("Single intercepts:")
    print(f"{first_intercept:.5}")
    print(f"{second_intercept:.5}")
    print(f"{third_intercept:.5}")

    plot_trend(
        np.log2(no_elements[:8]),
        times[:8],
        stds[:8],
        first_slope,
        first_intercept,
        color="red",
        ax=ax[0],
        label="Fitted Trend \n $2^1 - 2^8$",
    )
    plot_trend(
        np.log2(no_elements[8:15]),
        times[8:15],
        stds[8:15],
        second_slope,
        second_intercept,
        color="green",
        label="Fitted Trend \n $2^9 - 2^{15}$",
        fig=fig,
        ax=ax[0],
    )
    plot_trend(
        np.log2(no_elements[15:]),
        times[15:],
        stds[15:],
        third_slope,
        third_intercept,
        color="blue",
        fig=fig,
        ax=ax[0],
        label="Fitted Trend \n $2^{16} - 2^{20}$",
    )

    # Load two node case
    no_elements, times, stds, samples_per_element = read_data_from_csv(
        "assignment_1/pingpong_results_two_nodes.csv"
    )
    # Filter out first sample
    no_elements = no_elements[1:]
    times = times[1:] * 1e6  # Convert to microseconds
    stds = stds[1:] * 1e6  # Convert to microseconds
    times_filtered = times.copy()
    stds_filtered = stds.copy()
    times_filtered[8] = np.mean(samples_per_element[2**9][1:]) * 1e6
    stds_filtered[8] = np.std(samples_per_element[2**9][1:]) * 1e6
    # times_filtered[13] = np.mean(samples_per_element[2**14][1:]) * 1e6
    # stds_filtered[13] = np.std(samples_per_element[2**14][1:]) * 1e6

    # Compute trend
    first_slope, first_intercept, _, _, _ = stats.linregress(
        no_elements[:8], times_filtered[:8]
    )
    second_slope, second_intercept, _, _, _ = stats.linregress(
        no_elements[8:15], times_filtered[8:15]
    )
    third_slope, third_intercept, _, _, _ = stats.linregress(
        no_elements[15:], times_filtered[15:]
    )
    print("Two nodes slopes:")
    print(f"{first_slope:.5}")
    print(f"{second_slope:.5}")
    print(f"{third_slope:.5}")
    print("Two nodes intercepts:")
    print(f"{first_intercept:.5}")
    print(f"{second_intercept:.5}")
    print(f"{third_intercept:.5}")

    # Plot data
    plot_trend(
        np.log2(no_elements[:8]),
        times[:8],
        stds[:8],
        first_slope,
        first_intercept,
        color="red",
        # linestyle="--",
        fig=fig,
        ax=ax[1],
        # scatter_shape="s",
    )
    plot_trend(
        np.log2(no_elements[8:15]),
        times[8:15],
        stds[8:15],
        second_slope,
        second_intercept,
        color="green",
        fig=fig,
        ax=ax[1],
        # linestyle="--",
        # scatter_shape="s",
    )
    plot_trend(
        np.log2(no_elements[15:]),
        times[15:],
        stds[15:],
        third_slope,
        third_intercept,
        color="blue",
        fig=fig,
        ax=ax[1],
        # linestyle="--",
        # scatter_shape="s",
    )

    # Format Graphs
    ax[0].set_ylim(1, 1500)
    ax[1].set_ylim(1, 1500)
    ax[0].set_xlim(-1, 21)
    ax[1].set_xlim(-1, 21)
    ax[0].set_xlabel(r"$\log_2(\#\text{No of Elements})$")
    ax[1].set_xlabel(r"$\log_2(\#\text{No of Elements})$")
    ax[0].set_ylabel(r"$\log_{10}$ Time ($\mu s$)")
    ax[0].set_title("Single node")
    ax[1].set_title("Two nodes")
    fig.suptitle(r"Communication Delay Function ($2^{1} \rightarrow 2^{20}$)")

    ax[0].set_yscale("log")
    ax[1].set_yscale("log")
    ax[0].legend(
        # [
        #     r"Fitted Trend $2^1 - 2^8$",
        #     r"Fitted Trend $2^9 - 2^{15}$",
        #     r"Fitted Trend $2^{16} - 2^{20}$",
        #     r"Fitted Trend $2^1 - 2^8$ (Two compute nodes)",
        #     r"Fitted Trend $2^9 - 2^{15}$ (Two compute nodes)",
        #     r"Fitted Trend $2^{16} - 2^{20}$ (Two compute nodes)",
        # ]
    )
    ax[0].grid(
        which="both", axis="y", linestyle="--", alpha=0.4
    )  # Minor ticks on y-axis
    ax[1].grid(
        which="both", axis="y", linestyle="--", alpha=0.4
    )  # Minor ticks on y-axis
    plt.savefig("assignment_1/linear_trend_regular.png")
    plt.show()


if __name__ == "__main__":
    main()

""" Created on Wed Oct 16 11:56:28 2024
    @author: dcupolillo """

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Rectangle
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve


def plot_confusion_matrix(
        y_true,
        y_pred,
        classes=None,
        fontsize=10,
        cmap="Blues"):

    # Compute confusion matrix
    raw_cm = confusion_matrix(y_true, y_pred)
    cm_percentage = (
        raw_cm.astype('float') / raw_cm.sum(axis=1, keepdims=True))

    annot = []
    for i in range(raw_cm.shape[0]):
        row = []
        for j in range(raw_cm.shape[1]):
            percentage = f"{cm_percentage[i, j]:.2%}"
            raw = f"({raw_cm[i, j]})"
            row.append(f"{percentage}\n{raw}")
        annot.append(row)

    if classes is None:
        classes = ["Class 0", "Class 1"]

    # Plot the confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_percentage,
        ax=ax,
        annot=annot,
        fmt="",
        cmap=cmap,
        xticklabels=classes,
        yticklabels=classes,
        cbar=False,
        annot_kws={"size": fontsize}
    )
    ax.set_xlabel("Predicted", fontsize=fontsize)
    ax.set_ylabel("True", fontsize=fontsize)
    ax.tick_params('both', labelsize=fontsize)


def plot_sorted_traces_by_probability(
        data: np.ndarray,
        prediction: np.ndarray,
        label: np.ndarray,
        output: np.ndarray,
        colormap: str = 'viridis',
        increment: int = 5,
        n_zoomed_traces: int = 10,
        mid_zoom: tuple = None
) -> None:
    """
    Sorts test traces by predicted probability
    and visualizes them using a colormap.

    Args:
        data (list): List of test traces.
        prediction (list): List of binary predictions (0 or 1).
        label (list): List of ground truth labels.
        output (list): List of raw model output probabilities (0-1).
        colormap (str): Name of the matplotlib colormap to use.
    """
    if mid_zoom is not None and len(range(*mid_zoom)) != n_zoomed_traces:
        raise KeyError(
            "Insert a tuple that matches the length of `n_zoomed_traces`.")

    fig = plt.figure(
        figsize=(5, 8))
    spec = fig.add_gridspec(
        ncols=3,
        nrows=3,
        width_ratios=[.5, 0.8, .08])
    plt.subplots_adjust(
        left=0.15,
        right=0.9,
        top=0.95,
        bottom=0.1,
        wspace=0.4,
        hspace=0.4)

    ax_main = fig.add_subplot(spec[:, 0])
    ax_zoom1 = fig.add_subplot(spec[0, 1])
    ax_zoom2 = fig.add_subplot(spec[1, 1])
    ax_zoom3 = fig.add_subplot(spec[2, 1])
    cbar_ax = fig.add_subplot(spec[:3, 2])

    offset = 0

    # Convert lists to numpy arrays for sorting
    output = np.concatenate(output).ravel()  # Convert to 1D array
    data = np.concatenate(data)  # Convert to numpy array (batch, 1, time)
    labels = np.concatenate(label).ravel()

    period = 1 / 16
    timestamps = np.arange(
        0,
        (period * len(data[0][0])),
        period)

    zscore_scale = 5

    # **Sort indices by predicted probability (ascending)**
    sorted_indices = np.argsort(output)

    # **Normalize probability values between 0 and 1 for colormap**
    norm = plt.Normalize(vmin=0, vmax=1)
    cmap = cm.get_cmap(colormap)

    n_ticks = 2
    yticks = np.linspace(0, len(data) * increment, n_ticks)
    yticks_labels = np.linspace(0, len(data), n_ticks, dtype=int)

    # **Plot sorted traces with color mapping**
    for idx in sorted_indices:
        trace = data[idx][0]  # Extract single trace
        prob = output[idx]  # Get corresponding probability

        color = cmap(norm(prob))  # Map probability to color
        ax_main.plot(
            timestamps,
            trace + offset,
            color=color,
            alpha=0.8,
            lw=0.8,
            clip_on=False)

        offset += increment  # Move trace up

    ax_main.set_yticks(yticks)
    ax_main.set_yticklabels(yticks_labels)
    ax_main.set_ylim(0, yticks[-1])
    ax_main.set_xlim(0, timestamps[-1])
    ax_main.spines[["top", "right"]].set_visible(False)
    ax_main.spines[["left"]].set_position(("outward", 20))

    ax_main.set_xlabel("Time (s)")
    ax_main.set_ylabel("Test traces", labelpad=-10)

    top_offset = 0
    mid_offset = 0
    bottom_offset = 0

    top_zoom = (len(sorted_indices) - n_zoomed_traces)
    if mid_zoom is None:
        mid_zoom = (
            (len(sorted_indices) // 2) - (n_zoomed_traces // 2),
            (len(sorted_indices) // 2) + (n_zoomed_traces // 2))
    bottom_zoom = n_zoomed_traces

    rect_x_start, rect_x_end = timestamps[0] - 0.2, timestamps[-1] + 0.2
    top_y = (top_zoom * increment, (top_zoom + n_zoomed_traces) * increment)
    mid_y = (mid_zoom[0] * increment, mid_zoom[1] * increment)
    bottom_y = (0, bottom_zoom * increment)
    rect_kwargs = dict(
        edgecolor="red",
        facecolor="none",
        linewidth=1,
        clip_on=False,
        zorder=2)

    ax_main.add_patch(
        Rectangle((rect_x_start, top_y[0]),
                  rect_x_end - rect_x_start,
                  top_y[1] - top_y[0],
                  **rect_kwargs))
    ax_main.add_patch(
        Rectangle((rect_x_start, mid_y[0]),
                  rect_x_end - rect_x_start,
                  mid_y[1] - mid_y[0],
                  **rect_kwargs))
    ax_main.add_patch(
        Rectangle((rect_x_start, bottom_y[0]),
                  rect_x_end - rect_x_start,
                  bottom_y[1] - bottom_y[0],
                  **rect_kwargs))

    for n, (top_idx, mid_idx, bottom_idx) in enumerate(zip(
            sorted_indices[top_zoom:],
            sorted_indices[mid_zoom[0]:mid_zoom[1]],
            sorted_indices[:bottom_zoom])):

        top_trace = data[top_idx][0]  # Extract single trace
        mid_trace = data[mid_idx][0]
        bottom_trace = data[bottom_idx][0]

        top_prob = output[top_idx]  # Get corresponding probability
        mid_prob = output[mid_idx]
        bottom_prob = output[bottom_idx]

        top_true_label = int(labels[top_idx])
        mid_true_label = int(labels[mid_idx])
        bottom_true_label = int(labels[bottom_idx])

        top_color = cmap(norm(top_prob))  # Map probability to color
        mid_color = cmap(norm(mid_prob))
        bottom_color = cmap(norm(bottom_prob))

        ax_zoom1.plot(
            timestamps,
            top_trace + top_offset,
            color=top_color,
            alpha=0.8)
        ax_zoom2.plot(
            timestamps,
            mid_trace + mid_offset,
            color=mid_color,
            alpha=0.8)
        ax_zoom3.plot(
            timestamps,
            bottom_trace + bottom_offset,
            color=bottom_color,
            alpha=0.8)

        ax_zoom1.text(
            timestamps[-1] + 0.2, top_offset, f"{top_true_label}",
            verticalalignment='center', fontsize=10,
            color='red', clip_on=False)
        ax_zoom2.text(
            timestamps[-1] + 0.2, mid_offset, f"{mid_true_label}",
            verticalalignment='center', fontsize=10,
            color='red', clip_on=False)
        ax_zoom3.text(
            timestamps[-1] + 0.2, bottom_offset, f"{bottom_true_label}",
            verticalalignment='center', fontsize=10,
            color='red', clip_on=False)

        for n_ax, zoomed_ax in enumerate([ax_zoom1, ax_zoom2, ax_zoom3]):
            zoomed_ax.axvline(1.0, color='red', lw=0.5, alpha=0.6)
            zoomed_ax.set_xlim(0, 4)
            zoomed_ax.set(yticks=[])
            zoomed_ax.tick_params("y", length=0)

            zoomed_ax.vlines(
                x=3.7,
                ymin=0,
                ymax=zscore_scale,
                lw=0.5,
                color="black",
                clip_on=False)

            if n_ax == 2:
                zoomed_ax.spines[["top", "right", "left"]].set_visible(False)
                zoomed_ax.set_xlabel("Time (s)")

            else:
                zoomed_ax.spines[:].set_visible(False)
                zoomed_ax.tick_params("x", length=0)
                zoomed_ax.set(xticks=[])

        top_offset += increment
        mid_offset += increment
        bottom_offset += increment

    ax_zoom1.text(
        x=3.8,
        y=zscore_scale / 2,
        s=f"{zscore_scale} z-score",
        rotation=90,
        verticalalignment="center",
        horizontalalignment="left",
        color="black",
        clip_on=False
    )

    # **Colorbar for Probability Scale**
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar_ax.set_yticks([0, 1])
    cbar_ax.set_yticklabels([0, 1])
    cbar.set_label("Predicted Probability", labelpad=-5)


def plot_trial_metrics(
        trial_curves: dict,
        metric_keys: list
) -> None:
    """
    Plot training curves across trials.

    Parameters
    ----------
    trial_curves : dict
        Dictionary of trial results indexed by trial number.
    metric_keys : list
        List of metric names to plot (e.g., "train_loss", "validation_f1").
    """

    if not trial_curves or not metric_keys:
        return

    num_trials = len(trial_curves)
    rows = int(num_trials**0.5)
    cols = (num_trials // rows) + (num_trials % rows > 0)

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(15, 10),
        constrained_layout=True)

    axes = axes.flat
    fig.suptitle("Metric Curves per Trial")

    for i, (trial_id, trial_data) in enumerate(trial_curves.items()):
        ax = axes[i]
        for key in metric_keys:
            if key in trial_data and isinstance(trial_data[key], list):
                ax.plot(trial_data[key], label=key)
        ax.set(title=f"Trial {trial_id}", xlabel="Epochs", ylabel="Score")
        ax.grid(True)
        if i == 0:
            ax.legend()

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])


def plot_pr_curves(trial_curves: dict) -> None:
    """
    Plot PR curves and AUC-PR for trials with test labels and predictions.

    Parameters
    ----------
    trial_curves : dict
        Dictionary containing per-trial test labels and predictions.
    """
    filtered_trials = {
        trial_id: data for trial_id, data in trial_curves.items()
        if "test_labels" in data and "test_predictions" in data
    }

    if not filtered_trials:
        print("No trials with test_labels and test_predictions found.")
        return

    num_trials = len(filtered_trials)
    rows = int(num_trials ** 0.5)
    cols = (num_trials // rows) + (num_trials % rows > 0)

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(15, 10),
        constrained_layout=True,
        sharex=True,
        sharey=True
    )
    axes = axes.flat

    for i, (trial_id, trial_data) in enumerate(filtered_trials.items()):
        y_true = np.array(trial_data["test_labels"])
        y_pred = np.array(trial_data["test_predictions"])

        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        auc_pr = average_precision_score(y_true, y_pred)

        ax = axes[i]
        ax.plot(recall, precision, label=f"AUC-PR: {auc_pr:.3f}")
        ax.set(title=f"Trial {trial_id}", xlabel="Recall", ylabel="Precision")
        ax.grid(True)
        ax.legend()

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Precision-Recall Curves Across Trials")


def plot_roc_curves(trial_curves: list):
    """
    Plot ROC curves and AUC-ROC for trials with test labels and predictions.

    Parameters
    ----------
    trial_curves : dict
        Dictionary containing per-trial test labels and predictions.
    """

    num_trials = len(trial_curves)
    rows = int(num_trials ** 0.5)
    cols = (num_trials // rows) + (num_trials % rows > 0)

    fig, axes = plt.subplots(
        rows,
        cols,
        sharex=True,
        sharey=True,
        figsize=(15, 10),
        constrained_layout=True)
    axes = axes.flatten()

    for i, (trial_id, trial_data) in enumerate(trial_curves.items()):
        ax = axes[i]

        # Extract ROC curve data
        fpr = trial_data["fpr"]
        tpr = trial_data["tpr"]
        roc_auc = trial_data["roc_auc"]

        # Plot ROC curve
        ax.plot(fpr, tpr, color="blue", lw=2, label=f"AUC: {roc_auc:.2f}")
        ax.plot([0, 1], [0, 1], color="gray", linestyle="dashed")

        ax.set(
            title=f"Trial {trial_id}",
            xlabel="False Positive Rate",
            ylabel="True Positive Rate")

        ax.legend()
        ax.grid(True)

    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

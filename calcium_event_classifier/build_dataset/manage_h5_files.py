""" Created on Mon Oct 28 17:23:20 2024
    @author: dcupolillo """

import flammkuchen as fl
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


# %% Take only traces labeled as '1'


def filter_h5(
        h5_path: str or Path,
        label: int
) -> dict:
    """
    Filter a given binary dataset according to an input class.

    Parameters
    ----------
    h5_file : str or Path
        The input dataset.
    label : int
        The class to select, either 0 or 1.

    Returns
    -------
    dict
        The filtered dictionary.

    """
    ok_labels = [0, 1]

    if label not in ok_labels:
        raise ValueError(f"Incorrect label {label}. Should be {ok_labels}.")

    if not isinstance(h5_path, (str, Path)):
        raise TypeError("Input must be a string or a Path object.")

    h5_file = fl.load(h5_path)

    if not isinstance(label, int):
        raise TypeError("Label must be an integer.")

    if 'label' not in h5_file or 'zscore' not in h5_file:
        raise KeyError("Input file must contain 'label' and 'zscore' keys.")

    labels = h5_file['label']
    z_scores = h5_file['zscore']

    filtered_indices = (labels == label)

    filtered_labels = labels[filtered_indices]
    filtered_z_scores = z_scores[filtered_indices]

    h5_dictionary = {
        'label': filtered_labels,
        'zscore': filtered_z_scores
    }

    return h5_dictionary


# Usage
file_to_load = Path(r"test_dataset_250530.h5")
output_filename = Path(r"filtered_dataset_label1_250530.h5")

h5_dictionary = filter_h5(file_to_load, 1)
fl.save(output_filename, h5_dictionary)


# %% Merge h5 files


def merge_h5(
        h5_path_1: str or Path,
        h5_path_2: str or Path
) -> dict:
    """
    Concatenates two given datasets.

    Parameters
    ----------
    h5_file_1 : str or Path
        The first dataset.
    h5_file_2 : str or Path
        The second dataset.

    Returns
    -------
    dict
        The merged dictionary from the two files.
    """
    if not (isinstance(h5_path_1, (str, Path)) and
            isinstance(h5_path_2, (str, Path))):
        raise TypeError(
            "Both inputs must be a string or a Path object.")

    h5_file_1 = fl.load(h5_path_1)
    h5_file_2 = fl.load(h5_path_2)

    if 'label' not in h5_file_1 or 'label' not in h5_file_2:
        raise KeyError("Both files must contain 'label' key.")
    if 'zscore' not in h5_file_1 or 'zscore' not in h5_file_2:
        raise KeyError("Both files must contain 'zscore' key.")
    if 'dff' not in h5_file_1 or 'zscore' not in h5_file_2:
        raise KeyError("Both files must contain 'dff' key.")

    merged_labels = np.concatenate(
        [h5_file_1['label'],
         h5_file_2['label']])

    merged_z_scores = np.concatenate(
        [h5_file_1['zscore'],
         h5_file_2['zscore']])

    merged_dff = np.concatenate(
        [h5_file_1['dff'],
         h5_file_2['dff']])

    h5_dictionary = {
        'label': merged_labels,
        'zscore': merged_z_scores,
        'dff': merged_dff
    }

    return h5_dictionary


# Usage
file_path_1 = Path(r"C:/Users/dcupolillo/Projects/calcium_event_classifier/datasets/251016_test_dataset.h5_0only")
file_path_2 = Path(r"C:/Users/dcupolillo/Projects/calcium_event_classifier/datasets/251114_test_dataset.h5_1only")
output_filename = Path(r"mixed.h5")

h5_dictionary = merge_h5(file_path_1, file_path_2)
fl.save(output_filename, h5_dictionary)


# %%


def find_duplicates(
        data: dict
) -> list:
    """
    Finds the indices of duplicate traces within a dataset.
    Helper function to remove_duplicates().

    Parameters
    ----------
    h5_file : str or Path
        The input dataset.

    Returns
    -------
    list
        The list of duplicate indices.

    """
    if not isinstance(data, dict):
        raise TypeError("Input must be a dictionary.")

    z_scores = data['zscore']
    traces_as_tuples = [tuple(trace) for trace in z_scores]

    unique_traces = set()
    duplicates_index = []
    unique_index = []

    for i, trace in enumerate(traces_as_tuples):
        if trace in unique_traces:
            duplicates_index.append(i)
        else:
            unique_traces.add(trace)
            unique_index.append(i)

    print(f"Found {len(duplicates_index)} duplicates.")

    return unique_index


def remove_duplicates(
        h5_file: str or Path
) -> dict:
    """
    Remove duplicate traces within a dataset.
    Uses find_duplicates().

    Parameters
    ----------
    h5_file : str or Path
        The input dataset.

    Returns
    -------
    dict
        The resulting dictionary without duplicates.

    """
    data = fl.load(h5_file)
    # Get indices of unique traces
    unique_index = find_duplicates(data)

    # Filter the arrays using unique indices
    unique_labels = data['label'][unique_index]
    unique_z_scores = data['zscore'][unique_index]
    unique_dff = data['dff'][unique_index]

    # Create new dictionary with unique entries
    unique_data = {
        'label': unique_labels,
        'zscore': unique_z_scores,
        'dff': unique_dff
    }

    return unique_data


# Usage
file_path = Path(r"C:/Users/dcupolillo/Projects/calcium_event_classifier/examples/251114_test_dataset.h5")
unique_file = remove_duplicates(file_path)

fl.save("251114_dataset.h5", unique_file)


# %%


def reduce_class_entries(
        h5_file: str or Path,
        label: int,
        target_count: int,
        seed: int = 42
) -> dict:
    """
    Randomly downsample entries of a given class to a specified count.

    Parameters
    ----------
    h5_file : dict
        The input dataset dictionary.
    label : int
        The class label to downsample (0 or 1).
    target_count : int
        The number of entries to retain for the specified class.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict
        The updated dictionary with downsampled class entries.
    """

    if label not in [0, 1]:
        raise ValueError(f"Invalid label {label}. Expected 0 or 1.")

    if not isinstance(h5_file, (str, Path)):
        raise TypeError("Input must be a string or a Path object.")

    h5_file = fl.load(h5_file)

    if 'label' not in h5_file or 'zscore' not in h5_file:
        raise KeyError("Input file must contain 'label' and 'zscore' keys.")

    # if not isinstance(target_count, int) or target_count <= 0:
    #     raise ValueError("target_count must be a positive integer.")

    labels = h5_file['label']
    z_scores = h5_file['zscore']
    dff = h5_file['dff']

    # Identify indices by class
    target_indices = np.where(labels == label)[0]
    other_indices = np.where(labels != label)[0]

    if target_count > len(target_indices):
        raise ValueError(
            f"Requested {target_count} entries but only {len(target_indices)}"
            " available for label {label}.")

    # Random downsampling
    rng = np.random.default_rng(seed)
    selected_indices = rng.choice(
        target_indices, size=target_count, replace=False)

    # Combine with the other class
    final_indices = np.sort(
        np.concatenate([selected_indices, other_indices]))

    # Filter the arrays
    reduced_data = {
        'label': labels[final_indices],
        'zscore': z_scores[final_indices],
        'dff': dff[final_indices]
    }

    return reduced_data


file_path = Path(r"C:/Users/dcupolillo/Projects/calcium_event_classifier/datasets/251016_test_dataset.h5")
reduced = reduce_class_entries(file_path, label=1, target_count=0)
fl.save("C:/Users/dcupolillo/Projects/calcium_event_classifier/datasets/251016_test_dataset.h5_0only", reduced)


# %% Examine a dataset


def examine_h5(
        h5_file: str or Path
) -> None:
    """
    Display of class-separated traces.

    Parameters
    ----------
    h5_file : str or Path
        The input dataset.

    Returns
    -------
    None

    """
    if not isinstance(h5_file, (str, Path)):
        raise TypeError("Input must be a string or a Path object.")

    h5_file = fl.load(h5_file)

    if 'label' not in h5_file or 'zscore' not in h5_file:
        raise KeyError(
            "Input file must contain 'label' and 'zscore' keys.")

    if not (isinstance(h5_file['label'], np.ndarray) or
            not isinstance(h5_file['zscore'], np.ndarray)):
        raise TypeError(
            "Both 'label' and 'zscore' must be numpy arrays.")

    if len(h5_file['label']) != len(h5_file['zscore']):
        raise ValueError(
            "'label' and 'zscore' arrays must have the same length.")

    total_traces = len(h5_file['label'])
    zero_traces = (h5_file['label'] == 0).sum()
    one_traces = (h5_file['label'] == 1).sum()

    print(f"Number of entries: {total_traces}")
    print(f"Number of 0-labeled entries: {zero_traces}")
    print(f"Number of 1-labeled entries: {one_traces}")

    fig, axes = plt.subplots(
        2, 2,
        figsize=(3, 5),
        sharex=True, sharey=True,
        layout="constrained")

    axes[0, 0].set_ylabel(f"0-labeled:\n{zero_traces} / {total_traces}")
    axes[1, 0].set_ylabel(f"1-labeled:\n{one_traces} / {total_traces}")

    zero_label_zscores = [
        h5_file['zscore'][i] for i in range(total_traces)
        if h5_file['label'][i] == 0]

    one_label_zscores = [
        h5_file['zscore'][i] for i in range(total_traces)
        if h5_file['label'][i] == 1]

    zero_label_dFF = [
        h5_file['dff'][i] for i in range(total_traces)
        if h5_file['label'][i] == 0]

    one_label_dFF = [
        h5_file['dff'][i] for i in range(total_traces)
        if h5_file['label'][i] == 1]

    mean_zero_zscores = np.mean(zero_label_zscores, axis=0)
    mean_one_zscores = np.mean(one_label_zscores, axis=0)
    mean_zero_dFF = np.mean(zero_label_dFF, axis=0)
    mean_one_dFF = np.mean(one_label_dFF, axis=0)

    for i, label in enumerate(h5_file['label']):
        # Use data point indices as x-axis (consistent with GUI plotting)
        x_data = np.arange(len(h5_file['zscore'][i]))

        axes[(0, 0) if label == 0 else (1, 0)].plot(
            x_data,
            h5_file['zscore'][i],
            color='lightgray',
            alpha=0.6)
        axes[(0, 1) if label == 0 else (1, 1)].plot(
            x_data,
            h5_file['dff'][i],
            color='lightgray',
            alpha=0.6)

    # Overlay means
    for n, mean in enumerate([mean_zero_zscores, mean_one_zscores]):
        if len(mean) > 0:  # Only plot if there are traces for this label
            x_data = np.arange(len(mean))
            axes[n, 0].plot(
                x_data,
                mean,
                color='mediumblue' if n == 0 else "crimson",
                linewidth=2,
                label=f'Mean {n}-labeled')
    for n, mean in enumerate([mean_zero_dFF, mean_one_dFF]):
        if len(mean) > 0:  # Only plot if there are traces for this label
            x_data = np.arange(len(mean))
            axes[n, 1].plot(
                x_data,
                mean,
                color='mediumblue' if n == 0 else "crimson",
                linewidth=2,
                label=f'Mean {n}-labeled')

    # Set x-axis labels
    axes[0, 0].set_xlabel('Data Points')
    axes[0, 1].set_xlabel('Data Points')


# Usage
file_path = Path(r"C:/Users/dcupolillo/Projects/calcium_event_classifier/examples/251114_test_dataset.h5")
examine_h5(file_path)


# %% Balance 0 and 1 labels


def balance_labels(
        h5_file: str or Path
) -> dict:
    """
    Balances the majority class to the minority class.

    Parameters
    ----------
    h5_file : str or Path
        The input dataset.

    Returns
    -------
    dict
        The balanced dictionary.

    """
    if not isinstance(h5_file, (str, Path)):
        raise TypeError("Input must be a string or a Path object.")

    h5_file = fl.load(h5_file)

    if 'label' not in h5_file or 'zscore' not in h5_file:
        raise KeyError(
            "Input file must contain 'label' and 'zscore' keys.")

    if not (isinstance(h5_file['label'], np.ndarray) or
            not isinstance(h5_file['zscore'], np.ndarray)):
        raise TypeError(
            "Both 'label' and 'zscore' must be numpy arrays.")

    if len(h5_file['label']) != len(h5_file['zscore']):
        raise ValueError(
            "'label' and 'zscore' arrays must have the same length.")

    labels = h5_file['label']
    z_scores = h5_file['zscore']

    one_indices = np.where(labels == 1)[0]
    zero_indices = np.where(labels == 0)[0]

    num_ones = len(one_indices)

    if num_ones < len(zero_indices):
        sampled_zero_indices = np.random.choice(
            zero_indices, num_ones, replace=False)
    else:
        sampled_zero_indices = zero_indices

    balanced_indices = np.concatenate([one_indices, sampled_zero_indices])

    np.random.shuffle(balanced_indices)

    balanced_data = {
        'label': labels[balanced_indices],
        'zscore': z_scores[balanced_indices]
    }

    return balanced_data


# Usage
file_path = Path(r"zscore_labels.h5")
balanced_file = balance_labels(file_path)

fl.save("balanced_zscore_labels.h5", balanced_file)

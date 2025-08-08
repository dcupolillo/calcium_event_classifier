""" Created on Tue Jul 22 14:15:56 2025
    @author: dcupolillo """

from pathlib import Path
import flammkuchen as fl
import itertools
import torch
import calcium_event_classifier as cec
import matplotlib.pyplot as plt
import numpy as np

device = cec.set_device()

# Load model
model_path = Path(
    r"models/your_model_with_dff_replaced.pth")
classifier = cec.load_classifier(model_path)
classifier.to(device)
classifier.eval()

checkpoint = torch.load(model_path)

folder = Path(r"C:\Users\dcupolillo\Projects\spyne\data")

axons = "CA3"

fig, ax = plt.subplots(2, len([date for date in folder.iterdir()]))

for n, date in enumerate(folder.iterdir()):

    for cell_n in date.iterdir():

        zscores_data_path = (
            cell_n / f"time_series/adjusted_filter/zscores_{axons}.h5")

        dff_data_path = (
            cell_n / f"time_series/adjusted_filter/dFF_{axons}.h5")

        zscores = fl.load(zscores_data_path)
        zscores = list(itertools.chain(*zscores))

        dffs = fl.load(dff_data_path)
        dffs = list(itertools.chain(*dffs))

        n_samples = len(zscores)
        logits = torch.empty(n_samples, device='cpu')
        probs = torch.empty(n_samples, device='cpu')

        with torch.no_grad():

            for n_sweep, (zscore, dff) in enumerate(zip(
                    zscores, dffs)):

                logit, prob = cec.is_calcium_event(
                    zscore, dff, classifier, device)

                logits[n_sweep] = logit
                probs[n_sweep] = prob

        class_1 = []
        class_0 = []
        threshold = checkpoint["best_thresholds"][-1]

        for sweep_n, sweep in enumerate(zscores):

            if probs[sweep_n] > threshold:
                ax[1, n].plot(sweep, color="lightgray")
                class_1.append(sweep)
            else:
                ax[0, n].plot(sweep, color="lightgray")
                class_0.append(sweep)

        ax[1, n].plot(np.mean(class_1, axis=0), color="forestgreen")
        ax[0, n].plot(np.mean(class_0, axis=0), color="forestgreen")
        ax[1, n].set_title(
            f"Class 1, n={len(class_1)}\n"
            f"({(len(class_1) / (len(class_1) + len(class_0)) * 100):.2f}%)")
        ax[0, n].set_title(
            f"Class 0, n={len(class_0)}\n"
            f"({(len(class_0) / (len(class_1) + len(class_0)) * 100):.2f}%)")

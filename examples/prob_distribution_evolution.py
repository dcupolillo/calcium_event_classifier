""" Created on Tue Jul 22 13:43:37 2025
    @author: dcupolillo """

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

checkpoint_path = Path(
    r"models/your_model.pth")
checkpoint = torch.load(checkpoint_path)

valid_probs = checkpoint["valid_probs"]
epoch_labels = checkpoint["epoch_labels"]
epoch_features = checkpoint["epoch_features"]

num_epochs = len(epoch_labels)
assert num_epochs > 0

labels_per_epoch = [np.array(label) for label in epoch_labels]
probs_per_epoch = [torch.sigmoid(torch.tensor(logits)).cpu().numpy()
                   for logits in checkpoint["epoch_logits"]]

fig, ax = plt.subplots(figsize=(6, 4))
bins = np.linspace(0, 1, 40)
line0, = ax.plot([], [], label="No Event", color="blue", alpha=0.6)
line1, = ax.plot([], [], label="Event", color="orange", alpha=0.6)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel("Predicted probability")
ax.set_ylabel("Normalized count")
ax.legend()
title = ax.text(0.5, 1.05, "", size=12, ha="center", transform=ax.transAxes)

# --- Animation function ---
def animate(epoch_idx):
    ax.clear()
    y_true = labels_per_epoch[epoch_idx]
    y_prob = probs_per_epoch[epoch_idx]

    # Split per class
    prob0 = y_prob[np.array(y_true) == 0]
    prob1 = y_prob[np.array(y_true) == 1]

    ax.hist(prob0, bins=bins, alpha=0.6, label="No Event", color="blue", density=True)
    ax.hist(prob1, bins=bins, alpha=0.6, label="Event", color="orange", density=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 6)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Normalized count")
    ax.legend()
    title.set_text(f"Epoch {epoch_idx + 1}")
    ax.set_title(f"Epoch {epoch_idx + 1}")


ani = animation.FuncAnimation(fig, animate, frames=num_epochs, interval=500)

# Save or show
# ani.save("logit_histogram_evolution.mp4", dpi=150)
plt.show()

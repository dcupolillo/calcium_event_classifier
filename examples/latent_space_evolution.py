""" Created on Fri Jul 11 15:19:27 2025
    @author: dcupolillo """

import torch
from pathlib import Path
import numpy as np
import umap
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

path = Path(
    r"models/model_2channels.pth")
checkpoint = torch.load(path)

epoch_features = checkpoint["epoch_features"]
epoch_labels = checkpoint["epoch_labels"]

features_per_epoch = [feat.numpy() for feat in epoch_features]
labels_per_epoch = [label.numpy() for label in epoch_labels]

all_features = np.vstack(features_per_epoch)
all_labels = np.concatenate(labels_per_epoch)

reducer = umap.UMAP(
    n_neighbors=12,
    min_dist=1.,
    target_weight=0.7,
    metric="correlation",
    target_metric='categorical',
    random_state=42,)

reducer.fit(all_features, y=all_labels)
umap_per_epoch = [reducer.transform(f) for f in features_per_epoch]

# tsne = TSNE(n_components=2, perplexity=30, learning_rate=200)
# tsne_proj_all = tsne.fit_transform(all_features)

# tsne_per_epoch = []
# for epoch_features in features_per_epoch:
#     tsne = TSNE(
#         n_components=2, perplexity=30, learning_rate=200, random_state=42)
#     tsne_epoch = tsne.fit_transform(epoch_features)
#     tsne_per_epoch.append(tsne_epoch)


# Prepare animation functions
def init():
    sc.set_offsets(np.empty((0, 2)))
    sc.set_array(np.array([]))
    title.set_text("Epoch 0")
    return sc, title


def update(frame):
    # data = tsne_per_epoch[frame]
    data = umap_per_epoch[frame]
    labels = labels_per_epoch[frame]
    sc.set_offsets(data)
    sc.set_array(labels)
    title.set_text(f"Epoch {frame + 1}")
    return sc, title


# Plot animated UMAP
fig, ax = plt.subplots(figsize=(5, 4))
sc = ax.scatter([], [], c=[], cmap="coolwarm", alpha=0.7)
title = ax.set_title("")

ax.set(
   xlim=(-20, 20),
   ylim=(-20, 20),
   xticks=[-20, 0, 20],
   yticks=[-20, 0, 20],
   xlabel="UMAP1",
   ylabel="UMAP2")
ax.text(
   ax.get_xlim()[0],
   ax.get_ylim()[1],
   s="w/ Events",
   color="crimson",
   va="bottom", ha="left"
   )
ax.text(
    ax.get_xlim()[0],
    ax.get_ylim()[1],
    s="w/o Events",
    color="mediumblue",
    va="top", ha="left"
    )
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["bottom", "left"]].set_position(("outward", 10))

anim = FuncAnimation(
    fig, update,
    init_func=init,
    frames=len(umap_per_epoch),
    interval=100,
    blit=False)

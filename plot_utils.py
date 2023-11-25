import numpy as np
import matplotlib.pyplot as plt


def plot_leaves(embeddings, labels):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    circle = plt.Circle((0, 0), 1.0, color='r', alpha=0.1)
    ax.add_artist(circle)
    n = embeddings.shape[0]
    colors = get_colors(labels, color_seed=1234)
    ax.scatter(embeddings[:n, 0], embeddings[:n, 1], c=colors, s=50, alpha=0.6)
    ax.scatter(np.array([0]), np.array([0]), c='black')
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.axis("off")
    plt.show()
    return ax


def get_colors(y, color_seed=1234):
    """random color assignment for label classes."""
    np.random.seed(color_seed)
    colors = {}
    for k in np.unique(y):
        r = np.random.random()
        b = np.random.random()
        g = np.random.random()
        colors[k] = (r, g, b)
    return [colors[k] for k in y]
import numpy as np
import matplotlib.pyplot as plt


def mobius_add(x, y):
    """Mobius addition in numpy."""
    xy = np.sum(x * y, 1, keepdims=True)
    x2 = np.sum(x * x, 1, keepdims=True)
    y2 = np.sum(y * y, 1, keepdims=True)
    num = (1 + 2 * xy + y2) * x + (1 - x2) * y
    den = 1 + 2 * xy + x2 * y2
    return num / den


def mobius_mul(x, t):
    """Mobius multiplication in numpy."""
    normx = np.sqrt(np.sum(x * x, 1, keepdims=True))
    return np.tanh(t * np.arctanh(normx)) * x / normx


def geodesic_fn(x, y, nb_points=100):
    """Get coordinates of points on the geodesic between x and y."""
    t = np.linspace(0, 1, nb_points)
    x_rep = np.repeat(x.reshape((1, -1)), len(t), 0)
    y_rep = np.repeat(y.reshape((1, -1)), len(t), 0)
    t1 = mobius_add(-x_rep, y_rep)
    t2 = mobius_mul(t1, t.reshape((-1, 1)))
    return mobius_add(x_rep, t2)


def plot_geodesic(x, y, ax):
    """Plots geodesic between x and y."""
    points = geodesic_fn(x, y)
    ax.plot(points[:, 0], points[:, 1], color='black', linewidth=0.25, alpha=0.8)


def plot_leaves(tree, embeddings, labels, height):
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111)
    circle = plt.Circle((0, 0), 1.0, color='r', alpha=0.1)
    ax.add_artist(circle)
    for k in range(1, height + 1):
        circle_k = plt.Circle((0, 0), k / (height + 1), color='b', alpha=0.05)
        ax.add_artist(circle_k)
    n = embeddings.shape[0]
    colors = get_colors(labels, color_seed=1234)
    ax.scatter(embeddings[:n, 0], embeddings[:n, 1], c=colors, s=50, alpha=0.6)
    ax.scatter(np.array([0]), np.array([0]), c='black')
    for u, v in tree.edges():
        x = tree.nodes[u]['coords'].numpy()
        y = tree.nodes[v]['coords'].numpy()
        if tree.nodes[u]['is_leaf'] is False:
            ax.scatter(x[0], x[1], c='red', s=50, marker='s')
        if tree.nodes[v]['is_leaf'] is False:
            ax.scatter(y[0], y[1], c='red', s=50, marker='s')
        plot_geodesic(y, x, ax)
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
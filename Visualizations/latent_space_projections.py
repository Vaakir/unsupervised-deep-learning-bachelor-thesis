import umap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
from sklearn.preprocessing import StandardScaler


def plot_multiple_datasets(model, plot_functions, datasets, labels, colors, titles=None):
    """
    Plots multiple 2D projections for each dataset in a single figure.

    Parameters:
    - plot_functions: List of functions to be called for each projection (e.g., PCA, t-SNE, UMAP, etc.).
    - datasets: List of datasets to be encoded and projected. Each dataset corresponds to one function call.
    - labels: List of labels for each dataset in the legend.
    - colors: List of colors for each dataset's scatter plot.
    - titles: List of titles for each subplot (optional). If not provided, defaults to function names.
    """
    # Combine all datasets for consistent scaling
    combined_data = np.concatenate(datasets, axis=0)
    
    # Standardize the data
    scaler = StandardScaler()
    combined_data_scaled = scaler.fit_transform(combined_data.reshape(-1, combined_data.shape[-1])).reshape(combined_data.shape)
    
    # Get the model's encoded representation
    if model.VAE_model:
        z_mean, z_std, z = model.encoder.predict(combined_data_scaled)
        latent = z_mean
    else:
        latent = model.encoder.predict(combined_data_scaled)
    
    num_plots = len(plot_functions)
    num_rows = (num_plots + 2) // 3  # Arrange plots in rows, 3 per row
    fig, axes = plt.subplots(num_rows, 3, figsize=(18, 6 * num_rows))
    axes = axes.flatten()
    
    dim_red_2D = {
        "PCA": lambda latent: PCA(n_components=2).fit_transform(latent),
        "TSNE": lambda latent: TSNE(n_components=2, perplexity=5).fit_transform(latent), # random_state=42
        "UMAP": lambda latent: umap.UMAP(n_components=2).fit_transform(latent), # random_state=42
        "ISOMAP": lambda latent: Isomap(n_components=2).fit_transform(latent)
    }
    
    # Loop through each plot function and dataset to plot
    for i, plot_func in enumerate(plot_functions):
        ax = axes[i]
        plot_func_name = plot_func.__name__
        title = titles[i] if titles and len(titles) > i else plot_func_name
        points_2D = dim_red_2D[plot_func_name](latent)
        plot_proj(points_2D, datasets, colors, labels, ax, title)
        ax.legend()
    
    # Remove any unused subplots
    for j in range(num_plots, len(axes)):
        axes[j].axis('off')
    
    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()

def plot_proj(points_2D, datasets, colors, labels, ax, title):
    ax.scatter(points_2D[:, 0], points_2D[:, 1], color="gray", label="ALL", alpha=0.5)
    start_idx = 0
    for dataset, color, label in zip(datasets, colors, labels):
        end_idx = start_idx + len(dataset)
        ax.scatter(points_2D[start_idx:end_idx, 0], points_2D[start_idx:end_idx, 1], 
                   color=color, label=label, alpha=0.7)
        start_idx = end_idx

    ax.set_title(title)
    ax.legend()


"""
plot_multiple_datasets( 
    model=models_list4[1],
    plot_functions=[pPCA, pTSNE, pUMAP, pISOMAP, pENCODED],
    datasets=[AD, MCI, H],
    labels=["AD", "MCI", "H"],
    colors=["red", "green", "blue"],
    titles=["PCA", "t-SNE", "UMAP", "Isomap", "Encoded"]
)
"""
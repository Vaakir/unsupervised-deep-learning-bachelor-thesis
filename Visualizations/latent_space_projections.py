import umap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
from sklearn.preprocessing import StandardScaler


"""
WARNING!
SOME OF THESE FUNCTIONS SCALE ON THE DATASET GIVEN ONLY, AND NOT ON THE WHOLE, 
MEANING CLUSTERS OF AD, H, MCI COULD END UP SEPERATED EVEN THOUGH THEY AREN'T IN REALITY, 
I WILL FIX THIS IN THE FUTURE, BUT NOW I WILL HAVE MY WEEKEND.
"""

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
    num_plots = len(plot_functions)
    num_rows = (num_plots + 2) // 3  # Arrange plots in rows, 3 per row
    fig, axes = plt.subplots(num_rows, 3, figsize=(18, 6 * num_rows))
    
    # Flatten axes array for easier iteration
    axes = axes.flatten()
    
    # Loop through each plot function and dataset to plot
    for i, plot_func in enumerate(plot_functions):
        ax = axes[i]
        title = titles[i] if titles and len(titles) > i else plot_func.__name__
        plot_func(model, datasets, colors, labels, ax=ax, title=title)  # Use ax here
        ax.legend()
    # Remove any unused subplots
    for j in range(num_plots, len(axes)):
        axes[j].axis('off')
    
    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()

def pPCA(model, datasets, colors, labels, ax, title="PCA Projection"):
    """
    Plots the PCA projection of the encoded representation of the dataset.
    Applies scaling to all datasets combined, ensuring consistent transformation.

    Parameters:
    - model: The model used for encoding the dataset.
    - datasets: List of datasets to be encoded and projected.
    - colors: List of colors used for the scatter plots.
    - labels: List of labels for the datasets in the legend.
    - ax: The axis on which the plot should be drawn.
    - title: The title for the plot (default is 'PCA Projection').
    """
    # Combine all datasets for consistent scaling
    combined_data = np.concatenate(datasets, axis=0)
    scaler = StandardScaler()
    combined_data_scaled = scaler.fit_transform(combined_data.reshape(-1, combined_data.shape[-1])).reshape(combined_data.shape)
    
    # Get the encoded representation of the combined data
    if model.VAE_model:
        z_mean, z_std, z = model.encoder.predict(combined_data_scaled)
    else:
        z_mean = model.encoder.predict(combined_data_scaled)

    # Apply PCA on the encoded representation
    pca = PCA(n_components=2)
    points = pca.fit_transform(z_mean)

    ax.scatter(points[:, 0], points[:, 1], color="gray", label="ALL", alpha=0.5)
    
    # Plot the PCA projection for all datasets
    start_idx = 0
    for dataset, color, label in zip(datasets, colors, labels):
        end_idx = start_idx + len(dataset)
        ax.scatter(points[start_idx:end_idx, 0], points[start_idx:end_idx, 1], 
                   color=color, label=label, alpha=0.7)
        start_idx = end_idx

    ax.set_title(title)
    ax.legend()
    return points


# Function to plot t-SNE projection
def pTSNE(model, datasets, colors, labels, ax, perplexity=5, title="t-SNE Projection"):
    """
    Plots the t-SNE projection of the encoded representation of the dataset.

    Parameters:
    - model: The model used for encoding the dataset.
    - dataset: The dataset to be encoded and projected.
    - color: The color used for the scatter plot.
    - label: The label for the dataset in the legend.
    - perplexity: The perplexity parameter for t-SNE (default is 5).
    - title: The title for the plot (default is 't-SNE Projection').
    """
    combined_data = np.concatenate(datasets, axis=0)
    scaler = StandardScaler()
    combined_data_scaled = scaler.fit_transform(combined_data.reshape(-1, combined_data.shape[-1])).reshape(combined_data.shape)
    if model.VAE_model:
        z_mean, z_std, z = model.encoder.predict(combined_data_scaled)
    else:
        z_mean = model.encoder.predict(combined_data_scaled)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    
    
    r = tsne.fit_transform(z_mean)
    ax.scatter(r[:, 0], r[:, 1], color="gray", label="ALL", alpha=0.5)
    
    start_idx = 0
    for dataset, color, label in zip(datasets, colors, labels):
        end_idx = start_idx + len(dataset)
        ax.scatter(r[start_idx:end_idx, 0], r[start_idx:end_idx, 1], 
                   color=color, label=label, alpha=0.7)
        start_idx = end_idx


    ax.set_title(title)
    ax.legend()

def pUMAP(model, datasets, colors, labels, ax, title="UMAP Projection"):
    """
    Plots the UMAP projection of the encoded representation of the datasets.
    Applies scaling to all datasets combined, ensuring consistent transformation.

    Parameters:
    - model: The model used for encoding the dataset.
    - datasets: List of datasets to be encoded and projected.
    - colors: List of colors used for the scatter plots.
    - labels: List of labels for the datasets in the legend.
    - ax: The axis on which the plot should be drawn.
    - title: The title for the plot (default is 'UMAP Projection').
    """
    combined_data = np.concatenate(datasets, axis=0)
    scaler = StandardScaler()
    combined_data_scaled = scaler.fit_transform(combined_data.reshape(-1, combined_data.shape[-1])).reshape(combined_data.shape)
    if model.VAE_model:
        z_mean, z_std, z = model.encoder.predict(combined_data_scaled)
    else:
        z_mean = model.encoder.predict(combined_data_scaled)
    umap_model = umap.UMAP(n_components=2, random_state=42)
    umap_results = umap_model.fit_transform(z_mean)

    ax.scatter(umap_results[:, 0], umap_results[:, 1], color="gray", label="ALL", alpha=0.5)

    start_idx = 0
    for dataset, color, label in zip(datasets, colors, labels):
        end_idx = start_idx + len(dataset)
        ax.scatter(umap_results[start_idx:end_idx, 0], umap_results[start_idx:end_idx, 1], 
                   color=color, label=label, alpha=0.7)
        start_idx = end_idx

    ax.set_title(title)
    ax.legend()
    return umap_results

def pISOMAP(model, datasets, colors, labels, ax, title="Isomap Projection"):
    """
    Plots the Isomap projection of the encoded representation of the datasets.
    Applies scaling to all datasets combined, ensuring consistent transformation.

    Parameters:
    - model: The model used for encoding the dataset.
    - datasets: List of datasets to be encoded and projected.
    - colors: List of colors used for the scatter plots.
    - labels: List of labels for the datasets in the legend.
    - ax: The axis on which the plot should be drawn.
    - title: The title for the plot (default is 'Isomap Projection').
    """
    # Combine all datasets for consistent scaling
    combined_data = np.concatenate(datasets, axis=0)
    
    # Standardize the data
    scaler = StandardScaler()
    combined_data_scaled = scaler.fit_transform(combined_data.reshape(-1, combined_data.shape[-1])).reshape(combined_data.shape)
    
    # Get the model's encoded representation
    if model.VAE_model:
        z_mean, z_std, z = model.encoder.predict(combined_data_scaled)
    else:
        z_mean = model.encoder.predict(combined_data_scaled)
    
    isomap_model = Isomap(n_components=2)
    isomap_results = isomap_model.fit_transform(z_mean)

    ax.scatter(isomap_results[:, 0], isomap_results[:, 1], color="gray", label="ALL", alpha=0.5)

    # Plot each dataset with the corresponding color and label
    start_idx = 0
    for dataset, color, label in zip(datasets, colors, labels):
        end_idx = start_idx + len(dataset)
        ax.scatter(isomap_results[start_idx:end_idx, 0], isomap_results[start_idx:end_idx, 1], 
                   color=color, label=label, alpha=0.7)
        start_idx = end_idx

    ax.set_title(title)
    ax.legend()
    return isomap_results

def pENCODED(model, datasets, colors, labels, ax, title="Encoded Representation"):
    """
    Plots the encoded representation of the datasets in 2D space.
    Applies scaling to all datasets combined, ensuring consistent transformation.

    Parameters:
    - model: The model used for encoding the dataset.
    - datasets: List of datasets to be encoded and plotted.
    - colors: List of colors used for the scatter plots.
    - labels: List of labels for the datasets in the legend.
    - ax: The axis on which the plot should be drawn.
    - title: The title for the plot (default is 'Encoded Representation').
    """
    # Combine all datasets for consistent scaling
    combined_data = np.concatenate(datasets, axis=0)
    
    # Standardize the data
    scaler = StandardScaler()
    combined_data_scaled = scaler.fit_transform(combined_data.reshape(-1, combined_data.shape[-1])).reshape(combined_data.shape)
    
    # Get the model's encoded representation
    if model.VAE_model:
        z_mean, z_std, z = model.encoder.predict(combined_data_scaled)
    else:
        z_mean = model.encoder.predict(combined_data_scaled)
    
    # Plot the encoded representation for all datasets combined
    ax.scatter(z_mean[:, 0], z_mean[:, 1], color="gray", label="ALL", alpha=0.5)

    # Plot each dataset with the corresponding color and label
    start_idx = 0
    for dataset, color, label in zip(datasets, colors, labels):
        end_idx = start_idx + len(dataset)
        ax.scatter(z_mean[start_idx:end_idx, 0], z_mean[start_idx:end_idx, 1], 
                   color=color, label=label, alpha=0.7)
        start_idx = end_idx

    ax.set_title(title)
    ax.legend()
    return z_mean


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
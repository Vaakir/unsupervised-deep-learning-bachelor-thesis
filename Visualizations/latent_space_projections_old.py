import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
from sklearn.preprocessing import StandardScaler
import umap

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
        
        # Loop through datasets, labels, and colors
        for j, dataset in enumerate(datasets):
            plot_func(model, dataset, colors[j], labels[j], ax=ax, title=title)  # Use ax here
        ax.legend()
    # Remove any unused subplots
    for j in range(num_plots, len(axes)):
        axes[j].axis('off')
    
    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()

# Function to plot PCA projection
def pPCA(model, dataset, color, label, ax, title="PCA Projection"):
    """
    Plots the PCA projection of the encoded representation of the dataset.

    Parameters:
    - model: The model used for encoding the dataset.
    - dataset: The dataset to be encoded and projected.
    - color: The color used for the scatter plot.
    - label: The label for the dataset in the legend.
    - title: The title for the plot (default is 'PCA Projection').
    """
    if model.VAE_model:
        z_mean, z_std, z = model.encoder.predict(dataset)
    else:
        z_mean = model.encoder.predict(dataset)
    points = PCA(2).fit_transform(z_mean)
    ax.scatter(points[:, 0], points[:, 1], color=color, label=label)
    ax.set_title(title)

# Function to plot t-SNE projection
def pTSNE(model, dataset, color, label, ax, perplexity=5, title="t-SNE Projection"):
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
    scaler = StandardScaler()
    dataset_scaled = scaler.fit_transform(dataset.reshape(-1, dataset.shape[-1])).reshape(dataset.shape)
    if model.VAE_model:
        z_mean, z_std, z = model.encoder.predict(dataset_scaled)
    else:
        z_mean = model.encoder.predict(dataset_scaled)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    tsne_results = tsne.fit_transform(z_mean)
    ax.scatter(tsne_results[:, 0], tsne_results[:, 1], color=color, label=label)
    ax.set_title(title)

# Function to plot UMAP projection
def pUMAP(model, dataset, color, label, ax, title="UMAP Projection"):
    """
    Plots the UMAP projection of the encoded representation of the dataset.

    Parameters:
    - model: The model used for encoding the dataset.
    - dataset: The dataset to be encoded and projected.
    - color: The color used for the scatter plot.
    - label: The label for the dataset in the legend.
    - title: The title for the plot (default is 'UMAP Projection').
    """
    scaler = StandardScaler()
    dataset_scaled = scaler.fit_transform(dataset.reshape(-1, dataset.shape[-1])).reshape(dataset.shape)    
    if model.VAE_model:
        z_mean, z_std, z = model.encoder.predict(dataset_scaled)
    else:
        z_mean = model.encoder.predict(dataset_scaled)
    umap_model = umap.UMAP(n_components=2, random_state=42)
    umap_results = umap_model.fit_transform(z_mean)
    ax.scatter(umap_results[:, 0], umap_results[:, 1], color=color, label=label)
    ax.set_title(title)

# Function to plot Isomap projection
def pISOMAP(model, dataset, color, label, ax, title="Isomap Projection"):
    """
    Plots the Isomap projection of the encoded representation of the dataset.

    Parameters:
    - model: The model used for encoding the dataset.
    - dataset: The dataset to be encoded and projected.
    - color: The color used for the scatter plot.
    - label: The label for the dataset in the legend.
    - title: The title for the plot (default is 'Isomap Projection').
    """
    scaler = StandardScaler()
    dataset_scaled = scaler.fit_transform(dataset.reshape(-1, dataset.shape[-1])).reshape(dataset.shape)
    if model.VAE_model:
        z_mean, z_std, z = model.encoder.predict(dataset_scaled)
    else:
        z_mean = model.encoder.predict(dataset_scaled)
    isomap = Isomap(n_components=2)
    isomap_results = isomap.fit_transform(z_mean)
    ax.scatter(isomap_results[:, 0], isomap_results[:, 1], color=color, label=label)
    ax.set_title(title)

# Function to plot the encoded representation directly
def pENCODED(model, dataset, color, label, ax, title="Encoded Representation"):
    """
    Plots the encoded representation of the dataset in 2D space.

    Parameters:
    - model: The model used for encoding the dataset.
    - dataset: The dataset to be encoded and plotted.
    - color: The color used for the scatter plot.
    - label: The label for the dataset in the legend.
    - title: The title for the plot (default is 'Encoded Representation').
    """
    if model.VAE_model:
        z_mean, z_std, z = model.encoder.predict(dataset)
    else:
        z_mean = model.encoder.predict(dataset)
    ax.scatter(z_mean[:, 0], z_mean[:, 1], color=color, label=label)
    ax.set_title(title)

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
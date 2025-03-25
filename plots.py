import matplotlib.pyplot as plt
import pandas as pd
import torch

FEATURES = ["Heart Rate (BPM)", "Blood Oxygen Level (%)", "Step Count", "Sleep Duration (hours)", "Activity Level", "Stress Level"]

def plot_histograms(df: pd.DataFrame, title: str ="Histograms"):
    """Plot histograms of each feature (Heart Rate (BPM), Blood Oxygen Level (%), Step Count, Activity Level, Stress Level).

    Args:
        df (pd.DataFrame): data
        title (str, optional): main title. Defaults to "Histograms".
    """
    fig, axs = plt.subplots(2, 3, figsize=(10, 5))
    fig.suptitle(title)

    for i, ax in enumerate(axs.flat):
        ax.hist(df[FEATURES[i]], edgecolor="black", linewidth=0.5)
        ax.set_title(FEATURES[i])
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()


def plot_combined_histograms(clusters: dict, colors: list, title: str ="Combined Histograms"):
    """Plot combined histograms of all clusters.

    Args:
        clusters (dict): cluster based separated data
        colors (list): colors to differentiate clusters
        title (str, optional): main title. Defaults to "Combined Histograms".
    """
    # Combine all data of each feature (with respect to the clusters)
    combined_feature_data = {
        "Heart Rate (BPM)": [],
        "Blood Oxygen Level (%)": [],
        "Step Count": [],
        "Sleep Duration (hours)": [],    
        "Activity Level": [],
        "Stress Level": []
    }
    for i in range(len(clusters)):
        combined_feature_data["Heart Rate (BPM)"].append(list(clusters[i]["Heart Rate (BPM)"]))
        combined_feature_data["Blood Oxygen Level (%)"].append(list(clusters[i]["Blood Oxygen Level (%)"]))
        combined_feature_data["Step Count"].append(list(clusters[i]["Step Count"]))
        combined_feature_data["Sleep Duration (hours)"].append(list(clusters[i]["Sleep Duration (hours)"]))
        combined_feature_data["Activity Level"].append(list(clusters[i]["Activity Level"]))
        combined_feature_data["Stress Level"].append(list(clusters[i]["Stress Level"]))

    # Plot
    fig, axs = plt.subplots(2, 3, figsize=(10, 5))
    fig.suptitle(title)

    for i, ax in enumerate(axs.flat):
        ax.hist(combined_feature_data[FEATURES[i]], 
                color=colors, 
                edgecolor="black", 
                linewidth=0.5, 
                alpha=0.6, 
                align="mid", 
                stacked=True, 
                label=["Cluster 1", "Cluster 2", "Cluster 3"])
        ax.legend()
        ax.set_title(FEATURES[i])
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()


def plot_clustered_datapoints(data: torch.Tensor, clusters_num: int, labels: torch.Tensor, centroids: torch.Tensor, colors: list):
    """Plot clustered data points in 2D.

    Args:
        data (torch.Tensor): input data
        clusters_num (int): number of clusters
        labels (torch.Tensor): labels (order corresponding to data order)
        centroids (torch.Tensor): centroids (position)
        colors (list): colors to differentiate clusters
    """
    for i in range(clusters_num):
        # Plot data points
        plt.scatter(data[labels == i, 0], 
                    data[labels == i, 1], 
                    color=colors[i], 
                    alpha=0.6, 
                    label=f"Cluster {i+1}")
    # Plot centroids
    plt.scatter(centroids[:, 0], 
                centroids[:, 1], 
                color="black", 
                marker="X", 
                s=200, 
                label="Centroids")
    
    plt.title("Clustered Data Points")
    plt.xlabel("PCA component 1")
    plt.ylabel("PCA component 2")
    plt.legend()

    plt.show()


def plot_datapoints(data: torch.Tensor):
    """Plot data point in 2D.

    Args:
        data (torch.Tensor): input data
    """
    plt.scatter(data[:, 0], data[:, 1], alpha=0.6)
    plt.title("Data Points")
    plt.xlabel("PCA component 1")
    plt.ylabel("PCA component 2")
    plt.show()


def plot_datapoints_outliers(with_outliers: torch.Tensor, without_outliers: torch.Tensor, threshold: tuple=None):
    """Plot data points with and without outliers in 2D.

    Args:
        with_outliers (torch.Tensor): data with outliers
        without_outliers (torch.Tensor): data without outliers
        threshold (tuple, optional): outliers threshold (PCA component 1, PCA component 2). Defaults to None.
    """
    _, axs = plt.subplots(1, 2, figsize=(10, 5))
    # Plot with outliers
    axs[0].scatter(with_outliers[:, 0], with_outliers[:, 1], alpha=0.6)
    # Highlight thresholds
    if threshold:
        axs[0].axvline(x=threshold[0], color="r", linestyle="--", label="Outliers threshold")
        axs[0].axhline(y=threshold[1], color="r", linestyle="--")

    axs[0].legend()
    axs[0].set_title("PCA-reduced data - With outliers")
    axs[0].set_xlabel("PCA component 1")
    axs[0].set_ylabel("PCA component 2")
        
    # Plot without outliers
    axs[1].scatter(without_outliers[:, 0], without_outliers[:, 1], alpha=0.6)
    axs[1].set_title("PCA-reduced data - Without outliers")
    axs[1].set_xlabel("PCA component 1")
    axs[1].set_ylabel("PCA component 2")

    plt.tight_layout()
    plt.show()
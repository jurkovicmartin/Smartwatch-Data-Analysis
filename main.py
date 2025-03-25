import torch
from torch.utils.data import DataLoader
import pandas as pd

from clear import clear_dataset
from dataset import CustomDataset
from kmeans import KMeans
from pca import PCA
from plots import *
from measurements import statistical_analysis

def main():

    ### DATA LOADING AND PREPROCESSING

    dataset = clear_dataset("data/smartwatch.csv")

    dataset = CustomDataset(df=dataset)

    print(f"Data correlation matrix:\n{dataset.correlation_matrix}\n")
    print(f"{statistical_analysis(dataset.df)}\n")

    ### DIMENSIONALITY REDUCTION

    pca = PCA(2)
    pca.fit(dataset.data)
    reduced_data = pca.transform(dataset.data)
    
    reconstruction_error = pca.get_reconstruction_error(dataset.data, reduced_data)
    for i in range(dataset.data.shape[1]):
        print(f"{i+1}. feature variance: {torch.var(dataset.data[:, i])}, reconstruction error {reconstruction_error[i]}")

    # Normalization
    normalized_data = dataset.z_score_normalize(reduced_data)

    # Manually removing outliers
    threshold_0 = -5.0
    threshold_1 = 3.0

    df = pd.DataFrame(normalized_data.numpy())
    # Connect back IDs
    df["id"] = dataset.ids
    df = df[(df[0] > threshold_0) & (df[1] < threshold_1)]

    # df and dataset.df have the same indexing
    indices = df.index
    # Drop the same records in dataset.df as df
    dataset.df = dataset.df.loc[indices]

    # Extract IDs
    dataset.ids = df["id"]
    df = df.drop(columns=["id"])
    cleared_data = torch.tensor(df.values, dtype=torch.float32)
    # Update the dataset
    dataset.data = cleared_data

    print(f"\nRemoved outliers: {len(normalized_data) - len(cleared_data)}")
    print(f"Number of records without outliers: {len(cleared_data)}\n")

    plot_datapoints_outliers(normalized_data, dataset.data, (threshold_0, threshold_1))

    ### CLUSTERING
    
    # ! COLORS HAVE TO BE THE SAME LENGTH AS THE NUMBER OF CLUSTERS
    CLUSTERS_NUM = 3
    BATCH_SIZE = 5
    COLORS = ["red", "green", "blue"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device\n")

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

    kmeans = KMeans(clusters_num=CLUSTERS_NUM, device=device)
    kmeans.fit(dataloader)
    centroids = kmeans.centroids.to("cpu")

    labels = kmeans.forward(dataloader).to("cpu")

    plot_clustered_datapoints(dataset.data, CLUSTERS_NUM, labels, centroids, COLORS)

    ### CLUSTERS ANALYSIS

    for i in range(CLUSTERS_NUM):
        print(f"Number of records in {i+1}. cluster: {torch.sum(labels == i)}")

    # Put whole dataset together
    dataset.df["User ID"] = dataset.ids
    dataset.df["Cluster"] = labels

    # Splitting the dataset into datasets based on clusters
    clusters = {}
    for i in range(CLUSTERS_NUM):
        clusters[i] = dataset.df[dataset.df["Cluster"] == i]


    # Whole dataset
    print("\nDataset values:")
    # print(dataset.df.drop(columns=["User ID", "Cluster"]).agg(["min", "max", "mean", "median", "mode", "std", "var", "skew", "kurt"]))
    print(statistical_analysis(dataset.df, ["User ID", "Cluster"]))
    plot_histograms(dataset.df, "Histograms of whole dataset")

    # Clusters
    for i in range(CLUSTERS_NUM):
        print(f"\nCluster {i+1} values:")
        print(statistical_analysis(clusters[i], ["User ID", "Cluster"]))
        plot_histograms(clusters[i], f"Histograms of cluster {i+1}")


    plot_combined_histograms(clusters, COLORS)



if __name__ == "__main__":
    main()
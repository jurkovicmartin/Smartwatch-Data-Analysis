import torch

class KMeans:
    def __init__(self, clusters_num: int, device: torch.device):
        """Class that implements the K-Means algorithm.

        Args:
            clusters_num (int): number of clusters
            device (torch.device): computing device
        """
        self.clusters_num = clusters_num
        self.device = device
        self.centroids = None


    def fit(self, dataloader: torch.utils.data.DataLoader, max_epochs: int =100, learning_rate: float =0.1, tolerance: float =1e-6):
        """Learns the K-Means model. Initializes and updates centroids positions.

        Args:
            dataloader (torch.utils.data.DataLoader): dataloader with learning data
            max_epochs (int, optional): maximum number of learning epochs. Defaults to 100.
            learning_rate (float, optional): affects centroids position updating. Defaults to 0.1.
            tolerance (float, optional): centroids shift tolerance(if the maximum shift is less than this the learning ends). Defaults to 1e-6.
        """
        # Load the first batch
        first_batch = next(iter(dataloader)).to(self.device)

        # Initialize centroids (random points from the first batch)
        self.centroids = first_batch[torch.randint(0, first_batch.shape[0], (self.clusters_num,))]

        for epoch in range(max_epochs):
            cluster_counts = torch.zeros(self.clusters_num, device=self.device)
            # Store previous centroids (for convergence check)
            old_centroids = self.centroids.clone()

            for batch in dataloader:
                batch = batch.to(self.device)
                # Compute Euclidean distances between points from batch and centroids
                distances = torch.cdist(batch, self.centroids)
                # Select the closest centroid for each point
                cluster_labels = torch.argmin(distances, dim=1)

                # Update centroids
                for i in range(self.clusters_num):
                    # Points belonging to cluster k
                    cluster_points = batch[cluster_labels == i]
                    cluster_counts[i] += len(cluster_points) 
                    # If there are points in cluster k update the clusters centroid position
                    if len(cluster_points) > 0:
                        self.centroids[i] = (1 - learning_rate) * self.centroids[i] + learning_rate * cluster_points.mean(dim=0)

            # Check if there is any empty cluster
            for i in range(self.clusters_num):
                # If there is reinitialize the clusters centroid with a random point from the last batch
                if cluster_counts[i] == 0:
                    self.centroids[i] = batch[torch.randint(0, batch.shape[0], (1,))]
               
            # Check for convergence (compare maximum centroid shift with tolerance)
            if torch.max(torch.abs(self.centroids - old_centroids)) < tolerance:
                print(f"K-Means converged at epoch {epoch + 1}")
                return


    def forward(self, dataloader: torch.utils.data.DataLoader) -> torch.Tensor:
        """Forward pass of dataloader. Clusters the data.

        Args:
            dataloader (torch.utils.data.DataLoader): dataloader with data

        Returns:
            torch.Tensor: labels
        """
        labels = []
        for batch in dataloader:
            batch = batch.to(self.device)
            # Compute Euclidean distances between points from batch and centroids
            distances = torch.cdist(batch, self.centroids)
            # Select the closest centroid for each point
            labels.append(torch.argmin(distances, dim=1))
        # Concatenate labels into a single tensor
        return torch.cat(labels, dim=0)
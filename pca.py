import torch

class PCA:
    def __init__(self, n_components: int):
        """Class that implements Principal Component Analysis (PCA) for dimensionality reduction.

        Args:
            n_components (int): number of principal components (number of desired dimensions)
        """
        self.n_components = n_components
        self.components = None
        self.mean = None


    def fit(self, x: torch.Tensor):
        """Find the principal components (learn).

        Args:
            x (torch.Tensor): input data
        """
        # Mean of the data (of each feature)
        self.mean = x.mean(dim=0)
        # Center the data (mean of each feature is 0)
        x_centered = x - self.mean
        # Compute covariance matrix (represents correlation between features)
        cov = torch.mm(x_centered.t(), x_centered) / (x.shape[0] - 1)

        # Compute eigenvectors (represents direction of maximum variance)
        _, _, eigenvectors = torch.svd(cov)
        # Eigenvectors are already ordered so take the first n components
        self.components = eigenvectors[:, :self.n_components]


    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Transform input data (reduce dimensionality).

        Args:
            x (torch.Tensor): input data

        Returns:
            torch.Tensor: reduced data
        """
        # Project the data onto the principal components
        x_centered = x - self.mean
        return torch.mm(x_centered, self.components)
    

    def get_reconstruction_error(self, x: torch.Tensor, x_transformed: torch.Tensor) -> torch.Tensor:
        """Calculates Mean Squared Error (MSE) between original and reconstructed data.

        Args:
            x (torch.Tensor): original data
            x_transformed (torch.Tensor): PCA reduced data

        Returns:
            torch.Tensor: reconstruction error
        """
        # Reconstruct the data
        x_reconstructed = torch.mm(x_transformed, self.components.t()) + self.mean
        # Calculate mean squared error
        return torch.mean((x - x_reconstructed) ** 2, dim=0)
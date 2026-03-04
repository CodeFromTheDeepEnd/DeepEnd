import numpy as np

class TSNE:
    def __init__(self, n_components=2, perplexity=30.0, learning_rate=200.0,
                 n_iter=1000, early_exaggeration=12.0, early_exaggeration_iter=250):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.early_exaggeration = early_exaggeration
        self.early_exaggeration_iter = early_exaggeration_iter
        
    def map2d(self, X):
        """Main method: take high-dimensional data to 2D/3D-space"""
        # 1. Affinities in high-dimensional space (Gaussian kernel)
        P = self._compute_high_dim_affinities(X)
        
        # 2. Prepare low-dimensional data
        Y = self._initialize_embedding(X.shape[0])
        
        # 3. Early exaggeration: multiply affinities in the beginning
        P_exaggerated = P * self.early_exaggeration
        
        # 4. Do it
        Y = self._optimize(P_exaggerated, P, Y)
        
        return Y
    
    def _compute_high_dim_affinities(self, X):
        """
        Calcuate affinity matrix P in high-dimensional space.
        
        p_ij shows how "close" i and j are when using Gaussian kernel. 
        Notice, this is not a propability but a sort of geometrical weighing.
        """
        n = X.shape[0]
        
        # The squared distance matrix
        distances_sq = self._compute_squared_distances(X)
        
        # Compute affinities a(j|i) for each i
        # Use Gaussian kernel, with std \sigma_i from the perplexity
        P = np.zeros((n, n))
        for i in range(n):
            sigma_i = self._find_kernel_width(distances_sq[i], i, self.perplexity)
            P[i] = self._compute_conditional_affinities(distances_sq[i], i, sigma_i)
        
        # Symmetrize: p_ij = (a(j|i) + a(i|j)) / 2n
        P = (P + P.T) / (2 * n)
        
        # Stability
        P = np.maximum(P, 1e-12)
        
        return P
    
    def _compute_squared_distances(self, X):
        """Laske neliöetäisyysmatriisi euklidisessa metriikassa"""
        n = X.shape[0]

        # sum_X[i] = x_i[0]² + x_i[1]² + ... + x_i[n-1]² = ||x_i||²
        sum_X = np.sum(X**2, axis=1)

        # ||x_i - x_j||^2 = (x_i - x_j)·(x_i - x_j) = x_i·x_i + x_j·x_j - 2*x_i·x_j = ||x_i||^2 + ||x_j||^2 - 2*x_i·x_j
        # sum_X.reshape(-1, 1) is n-by-1 column vector with elements ||x_i||², sum_X.reshape(1, -1) is 1-by-n vector with elements ||x_j||², 
        # so their sum gives us the ||x_i||^2 + ||x_j||^2 part for all pairs (i,j) and then we just subtract 2 * X @ X.T to get the full distance matrix.
        D = sum_X.reshape(-1, 1) + sum_X.reshape(1, -1) - 2 * X @ X.T
        np.fill_diagonal(D, 0)
        return np.maximum(D, 0)
    
    def _find_kernel_width(self, distances_sq_i, i, target_perplexity):
        """
        Find σᵢ such that Perp(Pᵢ) = target_perplexity
        
        Perplexity defines the distance between neighbours. It is a hyperparamtere
        that controls how the local vs global dynamics pan out.
        """
        sigma_min = 1e-20
        sigma_max = 1e20
        sigma = 1.0
        
        for _ in range(50):
            affinities_i = self._compute_conditional_affinities(distances_sq_i, i, sigma)
            H = self._compute_shannon_entropy(affinities_i)
            perplexity = 2 ** H
            
            if abs(perplexity - target_perplexity) < 1e-5:
                break
                
            if perplexity > target_perplexity:
                sigma_max = sigma
                sigma = (sigma + sigma_min) / 2
            else:
                sigma_min = sigma
                sigma = (sigma + sigma_max) / 2
        
        return sigma
    
    def _compute_conditional_affinities(self, distances_sq_i, i, sigma):
        """
        Compute affinities a(j|i) with Gaussian kernel.
        
        We use normalized kernel, which puts more weight on close-by points.
        """
        # Gaussian kernel: exp(-||x_i - x_j||^2 / 2*sigma^2)
        affinities = np.exp(-distances_sq_i / (2 * sigma**2))
        affinities[i] = 0  # No affinity to self
        affinities = affinities / (np.sum(affinities) + 1e-8)  # Normalize
        return affinities
    
    def _compute_shannon_entropy(self, affinities):
        """
        Compute Shannon entropy from the affinity vector.
        
        Use perplexity, which is 2^H.
        """
        affinities = affinities[affinities > 0]
        return -np.sum(affinities * np.log2(affinities))
    
    def _initialize_embedding(self, n_samples):
        """
        Prepare low-dimensional data with a small random variance
        """
        return np.random.randn(n_samples, self.n_components) * 1e-4
    
    def _optimize(self, P_exaggerated, P, Y):
        """
        Optimize Y so that Q reflects P.      
        """
        momentum = 0.5
        final_momentum = 0.8
        momentum_switch_iter = 250
        
        Y_delta = np.zeros_like(Y)
        
        for iteration in range(self.n_iter):
            # The play with exaggeration
            P_current = P_exaggerated if iteration < self.early_exaggeration_iter else P
            
            if iteration == momentum_switch_iter:
                momentum = final_momentum
            
            # Affinities in the low-dimensional space
            Q = self._compute_low_dim_affinities(Y)
            
            # Gradient
            gradient = self._compute_gradient(P_current, Q, Y)
            
            # Update moment
            Y_delta = momentum * Y_delta - self.learning_rate * gradient
            Y = Y + Y_delta
            
            # Center (for visualization)
            Y = Y - np.mean(Y, axis=0)
            
            if (iteration + 1) % 100 == 0:
                kl_div = self._compute_kl_divergence(P_current, Q)
                print(f"Iteration {iteration + 1}: KL divergence = {kl_div:.4f}")
        
        return Y
    
    def _compute_low_dim_affinities(self, Y):
        """
        Compute affinities in the low-dimensional space using
        Student's t-distribution with one degree of freedom.
        
        t-distribution has heavier tails which gives the points
        far from each other more space, which in turn makes clusters
        better visible.
        """
        sum_Y = np.sum(Y**2, axis=1)
        distances_sq = sum_Y.reshape(-1, 1) + sum_Y.reshape(1, -1) - 2 * Y @ Y.T
        
        # Student's t-kernel: (1 + ||y_i - y_j||^2)^-1
        Q = 1 / (1 + distances_sq)
        np.fill_diagonal(Q, 0)
        
        # Normalize
        Q = Q / np.sum(Q)
        Q = np.maximum(Q, 1e-12)
        
        return Q
    
    def _compute_gradient(self, P, Q, Y):
        """
        Copute gradient for KL(P||Q).
        
        Gradient "pulls" points together, if P_ij > Q_ij 
        and pushes further, if P_ij < Q_ij.
        """
        n = Y.shape[0]
        
        PQ_diff = P - Q
        
        # The distances and t-distribution term
        sum_Y = np.sum(Y**2, axis=1)
        distances_sq = sum_Y.reshape(-1, 1) + sum_Y.reshape(1, -1) - 2 * Y @ Y.T
        inv_distances = 1 / (1 + distances_sq)
        np.fill_diagonal(inv_distances, 0)
        
        # Gradient from KL-divergence: 4 * sum_j (p_ij - q_ij) * (y_i - y_j) * (1 + d_ij^2)^-1
        gradient = np.zeros_like(Y)
        for i in range(n):
            diff = Y[i] - Y
            gradient[i] = 4 * np.sum(
                (PQ_diff[i] * inv_distances[i]).reshape(-1, 1) * diff,
                axis=0
            )
        
        return gradient
    
    def _compute_kl_divergence(self, P, Q):
        """
        Compute Kullback-Leibler-divergence KL(P||Q).
        
        Measures how much the affinities Q deviate from
        P. The smaller, the better.
        """
        return np.sum(P * np.log(P / Q))


# Example
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Generate structured data with 3 well-separated clusters
    np.random.seed(42)
    n_samples_per_cluster = 50
    
    # Cluster 1
    cluster1 = np.random.randn(n_samples_per_cluster, 50) * 0.3
    
    # Cluster 2 - shift only dimensions 2 onwards
    cluster2 = np.random.randn(n_samples_per_cluster, 50) * 0.3
    cluster2[:, 2:] += 10  # Shift only from 3rd dimension onwards
    
    # Cluster 3 - shift only dimensions 2 onwards  
    cluster3 = np.random.randn(n_samples_per_cluster, 50) * 0.3
    cluster3[:, 2:] -= 10  # Shift to opposite direction
    
    # Combine
    X = np.vstack([cluster1, cluster2, cluster3])
    labels = np.array([0] * n_samples_per_cluster + 
                      [1] * n_samples_per_cluster + 
                      [2] * n_samples_per_cluster)
    
    print(f"Input shape: {X.shape}")
    
    # Map to 2D with t-SNE
    tsne = TSNE(n_components=2, perplexity=30.0, learning_rate=200.0, 
                n_iter=2000, early_exaggeration=12.0)
    Y = tsne.map2d(X)
    
    print(f"Output shape: {Y.shape}")
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Original data (first 2 dims) - clusters NOT visible
    axes[0].scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6, s=50)
    axes[0].set_title('Original Data (first 2 dimensions)')
    axes[0].set_xlabel('Dimension 1')
    axes[0].set_ylabel('Dimension 2')
    axes[0].grid(True, alpha=0.3)
    
    # t-SNE projection - clusters visible
    axes[1].scatter(Y[:, 0], Y[:, 1], c=labels, cmap='viridis', alpha=0.6, s=50)
    axes[1].set_title('t-SNE Projection')
    axes[1].set_xlabel('t-SNE Component 1')
    axes[1].set_ylabel('t-SNE Component 2')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

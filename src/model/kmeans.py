



import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from cluster import ClusteringTemplate
from src.plotutils.plots import plot_elbow_curve, get_plots_gap, save_clustering_results
from src.plotutils.silhouette import plot_silhouette
import numpy as np
import os
from sklearn.metrics import silhouette_samples, silhouette_score

class KMeansClustering(ClusteringTemplate):
    def __init__(self):
        # super().__init__(self)
        super().__init__()

   
    def train(self, data,enable_vis):
        
        # Create directory to save results
        os.makedirs("results/kmeans/", exist_ok=True)
        clusters, scores = self.train_with_hyp(data)
        
        # Plot elbow curve
        plot_elbow_curve(scores, "kmeans", 1)

        if enable_vis is True:
            self.data = np.array(self.dimension_red(data))
            gap_stats = self.compute_gap_statistics(self.data)
            get_plots_gap(gap_stats, "kmeans", 1)
            range_n_clusters = [2, 3, 4, 5, 6] 
            plot_silhouette(self.data, range_n_clusters,"kmeans") 

        # Compute gap statistics
        gap_stats = range(2, 10)
        inertia = [scores[k] for k in range(2, 10)]
        silhouette_scores = [silhouette_score(data, clusters[k]) for k in range(2, 10)]
        david_bouldin_scores = [self.david_bouldin_score(data, clusters[k], kmeans.cluster_centers_) for k, kmeans in zip(range(2, 10), [KMeans(n_clusters=k).fit(data) for k in range(2, 10)])]
        
        # Save clustering results
        save_clustering_results("results/kmeans/clustering_results.csv", range(2, 10), scores, inertia, gap_stats, silhouette_scores, david_bouldin_scores)


        



        
        return clusters, scores

    def train_with_hyp(self, data):
        """
        Train KMeans with hyperparameters and return clusters and scores.
        
        Parameters:
        - data: Input data matrix (n_samples, n_features)
        
        Returns:
        - clusters: Dictionary containing clusters for each number of clusters
        - scores: Dictionary containing scores for each number of clusters
        """
        clusters = {}
        scores = {}
        for k in range(2, 10):  # Range of clusters [1, 2, ..., 9]
            kmeans = KMeans(n_clusters=k).fit(data)
            clusters[k] = kmeans.labels_
            scores[k] = kmeans.score(data)
        return clusters, scores

    def compute_gap_statistics(self,X):
        """
        Compute gap statistics for KMeans clustering.
        
        Parameters:
        - X: Input data matrix (n_samples, n_features)
        - k_max: Maximum number of clusters to consider
        - n_references: Number of reference datasets to generate
        
        Returns:
        - gap_stats: Array containing gap statistics for each value of k
        """
        # within_cluster_dispersion = np.zeros(k_max)
        k_max = 10
        n_references = 10
        gap_stats = np.zeros(k_max)
        for k in range(1, k_max + 1):
            kmeans = KMeans(n_clusters=k).fit(X)
            within_cluster_dispersion = kmeans.inertia_
        
            reference_dispersion = 0
            for i in range(n_references):
                random_data = np.random.rand(*X.shape)
                kmeans = KMeans(n_clusters=k).fit(random_data)
                reference_dispersion += kmeans.inertia_
        
            gap_stats[k - 1] = np.mean(np.log(reference_dispersion)) - np.log(within_cluster_dispersion)
    
        return gap_stats


    
    
        
    def david_bouldin_score(self,data, labels, centers):
        """
        Compute the David Bouldin score for KMeans clustering.
        
        Parameters:
        - data: Input data matrix (n_samples, n_features)
        - labels: Cluster labels for each data point
        - centers: Cluster centers
        
        Returns:
        - db_score: David Bouldin score
        """
        n_clusters = len(centers)
        distances = np.zeros((data.shape[0], n_clusters))
        
        # Compute distances from each data point to each cluster center
        for i in range(n_clusters):
            distances[:, i] = np.linalg.norm(data - centers[i], axis=1)
        
        # Compute the scatter within clusters
        scatter_w = np.zeros(n_clusters)
        for i in range(n_clusters):
            scatter_w[i] = np.mean(distances[labels == i, i])
        
        # Compute the pairwise distances between cluster centers
        pairwise_distances = np.zeros((n_clusters, n_clusters))
        for i in range(n_clusters):
            for j in range(n_clusters):
                pairwise_distances[i, j] = np.linalg.norm(centers[i] - centers[j])
        
        # Compute the David Bouldin score
        db_score = 0
        for i in range(n_clusters):
            max_val = -np.inf
            for j in range(n_clusters):
                if j != i:
                    val = (scatter_w[i] + scatter_w[j]) / pairwise_distances[i, j]
                    if val > max_val:
                        max_val = val
            db_score += max_val
        
        db_score /= n_clusters
        return db_score


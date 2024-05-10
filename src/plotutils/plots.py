
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def save_clustering_results(filename, hyperparameters, scores, inertia, gap_stats, silhouette_scores, david_bouldin_scores):
    """
    Save clustering results into a CSV file.
    
    Parameters:
    - filename: Name of the CSV file to save
    - hyperparameters: List of hyperparameters used for clustering
    - scores: Dictionary containing scores for each number of clusters
    - inertia: List containing inertia values for each number of clusters
    - gap_stats: Array containing gap statistics for each value of k
    - silhouette_scores: List of silhouette scores for each number of clusters
    - david_bouldin_scores: List of David Bouldin scores for each number of clusters
    
    Returns:
    - None (saves the data into a CSV file)
    """

    print(len(hyperparameters),len(scores),len(inertia),len(gap_stats),len(silhouette_scores),len(david_bouldin_scores))
    # Create a DataFrame to store the data
    df = pd.DataFrame({
        "Hyperparameters": hyperparameters,
        "Scores": list(scores.values()),
        "Inertia": inertia,
        "Silhouette Score": silhouette_scores,
        "David Bouldin Score": david_bouldin_scores
    })
    
    # Save the DataFrame to a CSV file
    df.to_csv(filename, index=False)

def get_plots_gap(gap_stats,alg,hyp):
    optimal_k = np.argmax(gap_stats) + 1
    print("Optimal number of clusters:", optimal_k)
    plt.plot(range(1, len(gap_stats) + 1), gap_stats, marker='o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Gap statistic')
    plt.title('Gap statistics for KMeans clustering')
    plt.savefig(f"results/{alg}/gap_stats.png")




def plot_elbow_curve(scores,alg,hyp):
    """
    Plot the elbow curve using the scores obtained from KMeans clustering.
    
    Parameters:
    - scores: Dictionary containing scores for each number of clusters
    
    Returns:
    - None (plots the elbow curve)
    """
    # Extract the number of clusters and corresponding scores
    k_values = list(scores.keys())
    score_values = list(scores.values())
    
    # Plot the elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, score_values, marker='o', linestyle='-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Score')
    plt.title('Elbow Curve for KMeans Clustering')
    plt.grid(True)
    print(alg)
    plt.savefig(f"results/{alg}/elbow_curve.png")

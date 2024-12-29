# %% [code]
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

def get_tsne_projection(embeddings, visualize = False, n_components=2, random_state=42):
    tsne = TSNE(n_components=n_components, random_state=random_state)
    
    # Fit and transform the embeddings
    embeddings_tsne = tsne.fit_transform(embeddings)

    if visualize:
    # Create a scatter plot
        plt.figure(figsize=(10, 8))
        plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], alpha=0.7)
        plt.title('t-SNE Visualization of Masked Headlines Embeddings')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.grid(True)
        plt.savefig("tSNE Visualization.png")

    return embeddings_tsne

def get_dbscan_clustering(embeddings = None, embeddings_tsne = None, visualize = False, eps=0.15, min_samples=20):
    if embeddings_tsne is None:
        embeddings_tsne = get_tsne_projection(embeddings)
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)  # Adjust eps and min_samples as needed
    dbscan_labels = dbscan.fit_predict(embeddings)

    if visualize:
        plt.figure(figsize=(10, 8))
        plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], c=dbscan_labels, cmap='plasma', s=50)
        plt.title('DBSCAN Clustering on t-SNE Data')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.colorbar(label='Cluster Label')
        plt.savefig("DBSCAN Clustering.png")

    return dbscan_labels

def get_gmm_clustering(embeddings = None, embeddings_tsne = None, visualize = False, n_components=10, random_state=0):
    if embeddings_tsne is None:
        embeddings_tsne = get_tsne_projection(embeddings)

    gmm = GaussianMixture(n_components=n_components, random_state=random_state)  # Adjust n_components (number of clusters) as needed
    gmm_labels = gmm.fit_predict(embeddings)
    
    # Plot the GMM clustering result
    if visualize:
        plt.figure(figsize=(10, 8))
        plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], c=gmm_labels, cmap='plasma', s=50)
        plt.title('GMM Clustering on t-SNE Data')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.colorbar(label='Cluster Label')
        plt.savefig("GMM Clusters.png")
    
    return gmm_labels

def get_agglomerative_clustering(embeddings = None, embeddings_tsne = None, visualize = False, n_clusters=15):
    if embeddings_tsne is None:
        embeddings_tsne = get_tsne_projection(embeddings)

    agglo = AgglomerativeClustering(n_clusters=n_clusters)  # Adjust n_clusters as needed
    agglo_labels = agglo.fit_predict(embeddings)

    if visualize:
    # Plot the Agglomerative Clustering result
        plt.figure(figsize=(10, 8))
        plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], c=agglo_labels, cmap='plasma', s=50)
        plt.title('Agglomerative Clustering on t-SNE Data')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.colorbar(label='Cluster Label')
        plt.savefig("Agglomerative.png")

    return agglo_labels

def get_clustering_technique(clustering: str,n_components: int):
    clustering_config = {
        'gmm': {
            'class': GaussianMixture,
            'param_name': 'n_components'
        },
        'agglo': {
            'class': AgglomerativeClustering,
            'param_name': 'n_clusters'
        },
        'dbscan': {
            'class': DBSCAN,
            'param_name': 'min_samples'
        }
    }
    
    clustering = clustering.lower()
    if clustering not in clustering_config:
        raise ValueError(f"Unsupported clustering technique: {clustering}")
    
    config = clustering_config[clustering]
    params = {config['param_name']: n_components}
    
    return config['class'](**params)


def optimize_n_components(clustering = None, n_components_range = range(2,50), embeddings = None):
    silhouette_scores = []    
    for n_components in n_components_range:
        # Get a new clusterer instance with updated parameters for each iteration
        clusterer = get_clustering_technique(clustering, n_components)
        cluster_labels = clusterer.fit_predict(embeddings)
        # Calculate the Silhouette Score only if more than one cluster is detected
        if len(np.unique(cluster_labels)) > 1:
            silhouette_avg = silhouette_score(embeddings, cluster_labels)
            silhouette_scores.append(silhouette_avg)
            print(f'n_components: {n_components}, Silhouette Score: {silhouette_avg}')
    
        else:
            silhouette_scores.append(-1)  # If only one cluster, append -1 for consistency
    
    
    
    # Plot the Silhouette Score curve
    plt.figure(figsize=(10, 5))
    plt.plot(n_components_range, silhouette_scores, marker='o', linestyle='-')
    plt.title(f'Silhouette Scores for Different Numbers of {clustering.upper()} Components')
    plt.xlabel('Number of Components')
    plt.ylabel('Silhouette Score')
    plt.grid(True)
    plt.savefig("Num Components vs Silhouette.png")
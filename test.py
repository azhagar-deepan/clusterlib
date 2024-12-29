import numpy as np
import heapq
import warnings
import time
from tqdm import tqdm
from numba import jit
from scipy.sparse import csr_matrix

warnings.simplefilter(action='ignore', category=FutureWarning)


def ROCK(df_original, sample_size=30, k=2, threshold=0.2, representativeness_fraction=0.5, min_cluster_size_percent=0.1, MIN_NEIGHBORS=2, BATCH_SIZE=100):
    """ROCK Clustering Algorithm with optimized operations."""

    # Convert DataFrame to NumPy array and preprocess
    df = df_original.astype(str).values
    n_samples, n_features = df.shape

    # Preprocess data with column indices - fixed dtype handling
    df_processed = []
    for i in range(n_samples):
        row_processed = []
        for j in range(n_features):
            row_processed.append(f"{j}|{df[i,j]}")
        df_processed.append(row_processed)
    df = np.array(df_processed, dtype=object)

    def jaccard_similarity(point1, point2):
        """Calculate the Jaccard similarity between two categorical points."""
        set1 = set(point1)
        set2 = set(point2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union != 0 else 0

    def compute_adjacency_matrix(S, threshold):
        """Compute the adjacency matrix based on the similarity threshold."""
        n = len(S)
        adjacency_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):  # Only compute upper triangle
                if i != j:
                    similarity = jaccard_similarity(S[i], S[j])
                    if similarity >= threshold:
                        adjacency_matrix[i, j] = adjacency_matrix[j, i] = 1
                    else:
                        adjacency_matrix[i, j] = adjacency_matrix[j, i] = 0
        return adjacency_matrix

    def calculate_links(S, threshold):
        """Calculate the links between pairs of points using sparse matrix multiplication."""
        adjacency_matrix = compute_adjacency_matrix(S, threshold)
        sparse_adj = csr_matrix(adjacency_matrix)
        links_matrix = (sparse_adj @ sparse_adj).toarray()
        return links_matrix

    def calculate_goodness_measure(Ci, Cj, links_matrix, threshold):
        """Calculate the goodness measure between two clusters."""
        ni = len(Ci)
        nj = len(Cj)
        link_sum = sum(links_matrix[i, j] for i in Ci for j in Cj)
        z = 1 + 2 * (1 - threshold / (1 + threshold))
        return link_sum / ((ni + nj) ** z - ni ** z - nj ** z)

    def label_remaining_points(df, sampled_points, clusters, threshold, representativeness_fraction, min_cluster_size_percent, MIN_NEIGHBORS, BATCH_SIZE):
        """Label points efficiently using batched processing."""
        representative_points = {}
        total_size = len(df)

        # Pre-compute representatives
        for cluster_id, points in clusters.items():
            cluster_size = len(points)
            if cluster_size / total_size < min_cluster_size_percent:
                representatives = sampled_points[points]
            else:
                num_representatives = max(1, int(cluster_size * representativeness_fraction))
                representatives = sampled_points[points][np.random.choice(len(points), num_representatives, replace=False)]
            representative_points[cluster_id] = representatives

        cluster_labels = np.full(len(df), 'NA')  # Initialize with 'NA'

        # Process in batches
        for batch_start in tqdm(range(0, len(df), BATCH_SIZE), desc="Processing Elements"):
            batch_end = min(batch_start + BATCH_SIZE, len(df))
            batch_points = df[batch_start:batch_end]

            for idx, point in enumerate(batch_points):
                max_score = -float('inf')  # Initialize to negative infinity
                best_cluster = None

                for cluster_id, representatives in representative_points.items():
                    N_i = sum(1 for rep in representatives if jaccard_similarity(point, rep) > threshold)

                    if MIN_NEIGHBORS == -1:
                        # When MIN_NEIGHBORS is -1, calculate score for all points
                        score = N_i / ((len(representatives) + 1) ** ((1 - threshold) / (1 + threshold)))
                        if score > max_score:
                            max_score = score
                            best_cluster = cluster_id
                    else:
                        # Normal behavior when MIN_NEIGHBORS > -1
                        if N_i >= MIN_NEIGHBORS:
                            score = N_i / ((len(representatives) + 1) ** ((1 - threshold) / (1 + threshold)))
                            if score > max_score:
                                max_score = score
                                best_cluster = cluster_id

                if best_cluster is not None:
                    cluster_labels[batch_start + idx] = str(best_cluster)

        na_count = np.sum(cluster_labels == 'NA')
        if MIN_NEIGHBORS > -1:  # Only show NA count when using minimum neighbors
            print(f"\nPoints assigned to NA: {na_count} ({(na_count/len(df))*100:.2f}%)")

        return cluster_labels.tolist()

    sampled_points = df[np.random.choice(n_samples, sample_size, replace=False)]
    links_matrix = calculate_links(sampled_points, threshold)

    # Initialize data structures
    local_heaps = {i: [] for i in range(sampled_points.shape[0])}
    clusters = {i: [i] for i in range(sampled_points.shape[0])}

    # Build local heaps
    for i in range(sampled_points.shape[0]):
        for j in range(i + 1, sampled_points.shape[0]):
            if links_matrix[i, j] > 0:
                goodness = calculate_goodness_measure(clusters[i], clusters[j], links_matrix, threshold)
                heapq.heappush(local_heaps[i], (-goodness, j))
                heapq.heappush(local_heaps[j], (-goodness, i))

    # Build global heap
    global_heap = []
    for i in clusters:
        if local_heaps[i]:
            max_goodness, max_index = local_heaps[i][0]
            heapq.heappush(global_heap, (max_goodness, i, max_index))
        else:
            heapq.heappush(global_heap, (0, i, -1))

    # Main clustering loop
    while len(clusters) > k and global_heap:
        max_goodness, u, v = heapq.heappop(global_heap)

        if u not in clusters or v not in clusters:
            continue

        # Merge clusters
        w = max(clusters.keys()) + 1
        clusters[w] = clusters[u] + clusters[v]
        del clusters[u], clusters[v]

        # Update links matrix
        new_size = links_matrix.shape[0] + 1
        new_links = np.zeros((new_size, new_size))
        new_links[:-1, :-1] = links_matrix

        for x in clusters:
            if x != w:
                new_links[w, x] = new_links[u, x] + new_links[v, x]
                new_links[x, w] = new_links[w, x]

        links_matrix = new_links

        # Update local heaps
        local_heaps[w] = []
        for x in clusters:
            if x != w:
                goodness = calculate_goodness_measure(clusters[w], clusters[x], links_matrix, threshold)
                heapq.heappush(local_heaps[w], (-goodness, x))
                heapq.heappush(local_heaps[x], (-goodness, w))

        # Rebuild global heap
        global_heap = []
        for i in clusters:
            if local_heaps[i]:
                max_goodness, max_index = local_heaps[i][0]
                heapq.heappush(global_heap, (max_goodness, i, max_index))

    print("\nRunning final labeling on the entire dataset...\n")
    final_labels = label_remaining_points(df, sampled_points, clusters, threshold,
                                        representativeness_fraction, min_cluster_size_percent, MIN_NEIGHBORS, BATCH_SIZE)

    return final_labels

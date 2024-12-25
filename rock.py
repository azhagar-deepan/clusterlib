import pandas as pd
import numpy as np
import heapq

def jaccard_similarity(point1, point2):
    """Calculate the Jaccard similarity between two categorical points."""
    set1 = set(point1)
    set2 = set(point2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

def compute_adjacency_matrix(S, threshold):
    """Compute the adjacency matrix based on the similarity threshold."""
    n = S.shape[0]
    adjacency_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                similarity = jaccard_similarity(S.iloc[i], S.iloc[j])
                adjacency_matrix[i, j] = 1 if similarity > threshold else 0

    return adjacency_matrix

def calculate_links(S, threshold):
    """Calculate the links between pairs of points using adjacency matrix multiplication."""
    adjacency_matrix = compute_adjacency_matrix(S, threshold)
    links_matrix = np.dot(adjacency_matrix, adjacency_matrix)  # Multiply A with itself
    return links_matrix

def calculate_goodness_measure(Ci, Cj, links_matrix, threshold):
    """Calculate the goodness measure between two clusters Ci and Cj."""
    ni = len(Ci)  # Size of cluster Ci
    nj = len(Cj)  # Size of cluster Cj
    link_Ci_Cj = sum(links_matrix[i, j] for i in Ci for j in Cj)  # Cross links
    z = 1 + 2 * (1 - threshold / (1 + threshold))
    goodness = link_Ci_Cj / ((ni + nj) ** z - ni ** z - nj ** z)
    return goodness

def label_remaining_points(df, sampled_points, clusters, threshold):
    """Label all points in the dataset based on clusters formed from sampled points."""
    # Extract representative points (L_i) from each cluster
    representative_points = {}
    for cluster_id, points in clusters.items():
        representative_points[cluster_id] = sampled_points.iloc[points]

    # Initialize cluster assignments for all points
    cluster_labels = [-1] * len(df)  # -1 indicates unclustered initially

    for idx, point in df.iterrows():
        max_similarity = -1
        best_cluster = -1

        # Compare the current point to representatives of each cluster
        for cluster_id, representatives in representative_points.items():
            similarities = [
                jaccard_similarity(point, representative) for _, representative in representatives.iterrows()
            ]
            normalized_similarity = sum(similarities) / (len(representatives) + 1) ** ((1 - threshold) / (1 + threshold))

            if normalized_similarity > max_similarity:
                max_similarity = normalized_similarity
                best_cluster = cluster_id

        # Assign the point to the best cluster
        cluster_labels[idx] = best_cluster

    return cluster_labels

def ROCK(S, k, threshold, df):
    """ROCK Clustering Algorithm with labeling."""
    n = S.shape[0]  # Number of sampled points
    print(f"Number of points in S: {n}")
    print(f"Desired number of clusters: {k}")

    # Calculate links matrix
    links_matrix = calculate_links(S, threshold)

    # Initialize local heaps and global heap
    local_heaps = {i: [] for i in range(n)}
    clusters = {i: [i] for i in range(n)}  # Initial clusters

    # Build local heaps
    for i in range(n):
        for j in range(n):
            if links_matrix[i, j] > 0 and i != j:
                goodness = calculate_goodness_measure(clusters[i], clusters[j], links_matrix, threshold)
                heapq.heappush(local_heaps[i], (-goodness, j))  # Store as negative for max-heap simulation

    # Build global heap
    global_heap = []
    for i in range(n):
        if local_heaps[i]:
            max_goodness, max_index = local_heaps[i][0]
            heapq.heappush(global_heap, (max_goodness, i, max_index))  # Track max of each cluster

    # Main while-loop to merge clusters until k clusters remain
    while len(clusters) > k:
        if not global_heap:  # If global_heap is empty, break out
            break

        # Extract the cluster pair with max goodness from the global heap
        max_goodness, u, v = heapq.heappop(global_heap)  # Extract the best pair of clusters to merge

        # Skip if either u or v is no longer valid
        if u not in clusters or v not in clusters:
            continue

        # Merge clusters u and v into w
        w = max(clusters.keys()) + 1  # Create a new unique cluster ID
        clusters[w] = clusters[u] + clusters[v]  # Combine the two clusters
        del clusters[u], clusters[v]  # Remove old clusters

        # Update links for the merged cluster
        links_matrix = np.pad(links_matrix, ((0, 1), (0, 1)), mode='constant')
        for x in list(clusters.keys()):
            if x != w:
                links_matrix[w, x] = links_matrix[u, x] + links_matrix[v, x]
                links_matrix[x, w] = links_matrix[w, x]

                # Recalculate goodness measure for x with the merged cluster
                goodness = calculate_goodness_measure(clusters[x], clusters[w], links_matrix, threshold)
                heapq.heappush(local_heaps[x], (-goodness, w))

        # Create new local heap for w
        local_heaps[w] = []
        for x in list(clusters.keys()):
            if x != w:
                goodness = calculate_goodness_measure(clusters[w], clusters[x], links_matrix, threshold)
                heapq.heappush(local_heaps[w], (-goodness, x))

        # Rebuild global heap
        global_heap = []
        for i in clusters.keys():
            if local_heaps[i]:
                max_goodness, max_index = local_heaps[i][0]
                heapq.heappush(global_heap, (max_goodness, i, max_index))

    # Final clusters formed
    print("Final Clusters:", clusters)

    # Label all points in the dataset based on clusters formed from sampled points
    cluster_labels = label_remaining_points(df, S, clusters, threshold)

    return cluster_labels

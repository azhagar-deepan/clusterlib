import pandas as pd
import numpy as np
import heapq
import warnings
import time
import tqdm

warnings.simplefilter(action='ignore', category=FutureWarning)


def ROCK(df_original, sample_size=30, k=2, threshold=0.2, representativeness_fraction=0.5, min_cluster_size_percent=0.1):
    """ROCK Clustering Algorithm with one consolidated function."""
    
    df = df_original.copy() # Create a copy of the original DataFrame
    

    for colnum in range(len(df.columns)):
        # Ensure the column is of string type before applying the lambda function
        df.iloc[:, colnum] = df.iloc[:, colnum].astype(str) 
        df.iloc[:, colnum] = df.iloc[:, colnum].apply(lambda x: f"{colnum}|{x}")

         

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
                    adjacency_matrix[i, j] = 1 if similarity >= threshold else 0

        return adjacency_matrix

    def calculate_links(S, threshold):
        """Calculate the links between pairs of points using adjacency matrix multiplication."""
        adjacency_matrix = compute_adjacency_matrix(S, threshold)
        links_matrix = np.dot(adjacency_matrix, adjacency_matrix)  # Multiply A with itself
        #print(links_matrix) #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        return links_matrix

    def calculate_goodness_measure(Ci, Cj, links_matrix, threshold):
        """Calculate the goodness measure between two clusters Ci and Cj."""
        ni = len(Ci)  # Size of cluster Ci
        nj = len(Cj)  # Size of cluster Cj
        link_Ci_Cj = sum(links_matrix[i, j] for i in Ci for j in Cj)  # Cross links
        z = 1 + 2 * (1 - threshold / (1 + threshold))
        goodness = link_Ci_Cj / ((ni + nj) ** z - ni ** z - nj ** z)
        return goodness
    
    def label_remaining_points(df, sampled_points, clusters, threshold, representativeness_fraction, min_cluster_size_percent):
        """Label all points in the dataset based on clusters formed from sampled points."""
        # Extract representative points (L_i) from each cluster
        representative_points = {}
        total_size = len(df)
    
        for cluster_id, points in clusters.items():
            cluster_size = len(points)
            if cluster_size / total_size < min_cluster_size_percent:
                # Use all points in the cluster as representatives
                representatives = sampled_points.iloc[points]
            else:
                # Use a fraction of points in the cluster as representatives
                num_representatives = max(1, int(cluster_size * representativeness_fraction))  # Ensure at least one representative
                representatives = sampled_points.iloc[points].sample(n=num_representatives)
    
            representative_points[cluster_id] = representatives
    
        # Initialize cluster assignments for all points
        cluster_labels = [-1] * len(df)  # -1 indicates unclustered initially
    
        with tqdm.tqdm(total=df.shape[0], desc="Processing Elements") as pbar: 
          for idx, point in df.iterrows():
              max_score = -1
              best_cluster = -1
              pbar.update(1)
              # Compare the current point to representatives of each cluster
              for cluster_id, representatives in representative_points.items():
                  # Calculate N_i: number of neighbors in representative set for the current point
                  N_i = sum(1 for representative in representatives.iterrows() if jaccard_similarity(point, representative[1]) > threshold)
      
                  # Calculate the score using the provided formula
                  score = N_i / ((len(representatives) + 1) ** ((1 - threshold) / (1 + threshold)))
      
                  # Check if this score is the highest we've seen
                  if score > max_score:
                      max_score = score
                      best_cluster = cluster_id
      
              # Assign the point to the best cluster
              cluster_labels[idx] = best_cluster
            
        return cluster_labels    # Sample points from the DataFrame to simulate the set S
        
    sampled_points = df.sample(n=sample_size)
    
    # Calculate links matrix
    links_matrix = calculate_links(sampled_points, threshold)

    # Initialize local heaps and global heap
    local_heaps = {i: [] for i in range(sampled_points.shape[0])}
    clusters = {i: [i] for i in range(sampled_points.shape[0])}  # Initial clusters

    # Build local heaps
    for i in range(sampled_points.shape[0]):
        for j in range(sampled_points.shape[0]):
            if links_matrix[i, j] > 0 and i != j:
                goodness = calculate_goodness_measure(clusters[i], clusters[j], links_matrix, threshold)
                heapq.heappush(local_heaps[i], (-goodness, j))  # Store as negative for max-heap simulation

    #print(local_heaps) #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    
    # Build Global Heap with all clusters, including those with empty local heaps
    global_heap = []
    for i in clusters.keys():
        if local_heaps[i]:
            max_goodness, max_index = local_heaps[i][0]
            heapq.heappush(global_heap, (max_goodness, i, max_index))  # Track max of each cluster
        else:
            # Optionally add the cluster with a default value or handle it differently
            heapq.heappush(global_heap, (0, i, -1))  # Adding with a goodness of 0 or a placeholder
    #print(global_heap) #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

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
    #print("Final Clusters:", clusters)

    # Run final labeling on the entire dataset
    print("\nRunning final labeling on the entire dataset...\n")
    final_labels = label_remaining_points(df, sampled_points, clusters, threshold,
                                           representativeness_fraction, min_cluster_size_percent)

    # Add cluster labels to the DataFrame
    return final_labels

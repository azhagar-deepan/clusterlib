'''

Cluster analysis algorithm: ROCK

Based on article description:
 - S.Guha, R.Rastogi, K.Shim. ROCK: A Robust Clustering Algorithm for Categorical Attributes. 1999.

Implementation by Andrei Novikov (spb.andr@yandex.ru)

'''

import math


# support methods
##test
def ed(i,j):
    sumSquare = 0
    for k in range(len(i)):
        sumSquare += abs(i[k] - j[k])**2
    return math.sqrt(sumSquare)

def jaccardConi(a,b):
    intersect = [val for val in a if val in b]
    union_liist = list(set(a).union(set(b)))

    if len(intersect) == 0:
        return 0.0
    else:
        return len(intersect)*1.0/len(union_list)

def read_sample(filename):
    """ Return sample for cluster analysis. """
    file = open(filename,'r')
    sample = [[float(val) for val in line.split()] for line in file];

    file.close()
    return sample

def euclidean_distance(a,b):
    """ Return Euclidean Distance between point a and point b """
    sumSquare = 0
    for k in range(len(a)):
        sumSquare += abs(a[k] - b[k]) ** 2
    return math.sqrt(sumSquare)


def rock(data, eps, number_clusters, threshold = 0.5):
    
    degree_normalization = 1.0 + 2.0 * ( (1.0 - threshold) / (1.0 + threshold) );
    adjacency_matrix = create_adjacency_matrix(data, eps);
    clusters = [[index] for index in range(len(data))];
    
    while (len(clusters) > number_clusters):
        indexes = find_pair_clusters(clusters, adjacency_matrix, degree_normalization);
        
        if (indexes != [-1, -1]):
            clusters[indexes[0]] += clusters[indexes[1]];
            clusters.pop(indexes[1]);   # remove merged cluster.
        else:
            break;  # totally separated clusters have been allocated
    
    return clusters;



def create_adjacency_matrix(data, eps):  
    size_data = len(data);
    
    adjacency_matrix = [ [ 0 for i in range(size_data) ] for j in range(size_data) ];
    for i in range(0, size_data):
        for j in range(i + 1, size_data):
            distance = euclidean_distance(data[i], data[j]);
            ## jaccardCon
            if (distance <= eps):
                adjacency_matrix[i][j] = 1;
                adjacency_matrix[j][i] = 1;
    
    return adjacency_matrix;



def calculate_links(cluster1, cluster2, adjacency_matrix):
    
    number_links = 0;
    
    for index1 in cluster1:
        for index2 in cluster2:
            number_links += adjacency_matrix[index1][index2];
            
    return number_links;
  


def find_pair_clusters(clusters, adjacency_matrix, degree_normalization):    
    maximum_goodness = 0.0;
    cluster_indexes = [-1, -1];
    
    for i in range(0, len(clusters)):
        for j in range(i + 1, len(clusters)):
            goodness = calculate_goodness(clusters[i], clusters[j],
                                          adjacency_matrix, degree_normalization);
            if (goodness > maximum_goodness):
                maximum_goodness = goodness;
                cluster_indexes = [i, j];
    
    return cluster_indexes;          


def calculate_goodness(cluster1, cluster2, adjacency_matrix, degree_normalization):

    number_links = calculate_links(cluster1, cluster2, adjacency_matrix);
    devider = (len(cluster1) + len(cluster2)) ** degree_normalization - len(cluster1) ** degree_normalization - len(cluster2) ** degree_normalization;
    
    return (number_links / devider);


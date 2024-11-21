import copy
import numpy as np

from sortedcontainers import SortedList, SortedKeyList
from utils import tanimoto_coefficient, spherical_distance, euclidean_distance


class Cluster:
    def __init__(self, points, sorted_list=None):
        """
        Cluster contains a list of points that belong to it and a sorted list of (goodness measure, cluster with
        common neighbor) tuples called local heaps in the paper. cluster2tuple index allows to find tuples in
        constant time

        :param points: Points belonging to this cluster
        :param sorted_list: Local sorted_list of clusters with common neighbors
        """
        self.points = points
        self.sorted_list = SortedKeyList(key=Cluster._sorted_lists_key)
        self.cluster2tuple = {}
        if sorted_list is not None:
            self.sorted_list = sorted_list

    @staticmethod
    def _sorted_lists_key(tup):
        return tup[0], tup[1].points[0]

    def __lt__(self, other):
        """Comparing clusters according to goodness measure of their first sorted lists elements.
        If cluster is outlying and heap is empty than we want it to go down in the global sorted list/heap Q"""
        if len(self.sorted_list) != 0 and len(other.sorted_list) != 0:
            if self.sorted_list[0][0] == other.sorted_list[0][0]:
                return self.points[0] < other.points[0]
            return self.sorted_list[0][0] < other.sorted_list[0][0]
        return len(other.sorted_list) == 0

    def __eq__(self, other):
        return self.points[0] == other.points[0]

    def __hash__(self):
        return self.points[0]


class RockClustering:
    """http://theory.stanford.edu/~sudipto/mypapers/categorical.pdf"""
    """ Many variable names used in this implementation correspond to the original paper linked above,
    please take a look before you start reading the code. Instead of heaps this implementation uses sorted lists
    In case of lack of memory, change links matrix to sparse matrix (Scipy lil_matrix) but this will slow down 
    the algorithm."""

    def __init__(self, S: np.array, k: int, theta=0.5):
        """
        :param S: Data set to cluster
        :param k: Number of clusters to find
        :param theta: Theta parameter from the original paper, should be from 0 to 1
        """
        self.S = S
        self.k = k
        self.theta = theta
        f_of_theta = (1.0 - self.theta) / (1.0 + self.theta)
        self.goodness_exponent = 1 + 2 * f_of_theta  # Used for goodness of measure calculation
        self.links = self.compute_links()
        self.q = [Cluster([i]) for i in range(0, S.shape[0])]
        self.global_sorted_list = SortedList()  # In the paper it is Q
        self.clustering()

    def clustering(self) -> None:
        """Performs iterative cluster merging, O(n^3) worst case, can be n^2log(n) if you use linked lists to store a heap"""
        for index in range(0, self.S.shape[0]):
            # Build local heaps
            points_linked = np.nonzero(self.links[index, :])[0]
            for point in points_linked:
                heap_tuple = (self.goodness_measure(self.q[index], self.q[point]), self.q[point])
                self.q[index].sorted_list.add(heap_tuple)
                self.q[index].cluster2tuple[heap_tuple[1]] = heap_tuple

        self.global_sorted_list = SortedList(copy.copy(self.q))

        while len(self.global_sorted_list) > self.k:
            #  Merge clusters until you reach k clusters or local heaps are empty and ROCK can't continue
            if not self.global_sorted_list[0].sorted_list:
                print("Could not find more clusters to merge. Stopped at {} clusters".format(
                    len(self.global_sorted_list)))
                break
            u = self.global_sorted_list.pop(0)
            v = u.sorted_list[0][1]
            self.global_sorted_list.discard(v)

            w = Cluster(sorted([*u.points, *v.points]))
            self.q[max(u.points[0], v.points[0])] = None
            self.q[w.points[0]] = w
            nbr_clusters = set([*[j for i, j in u.sorted_list], *[j for i, j in v.sorted_list]])
            nbr_clusters.remove(u)
            nbr_clusters.remove(v)

            for x in nbr_clusters:
                #  Update local heaps of neighbors of merged cluster u and v
                self.global_sorted_list.remove(x)
                try:
                    local_tuple = x.cluster2tuple[u]
                    x.sorted_list.discard(local_tuple)
                    del x.cluster2tuple[u]
                except KeyError:
                    pass

                try:
                    local_tuple = x.cluster2tuple[v]
                    x.sorted_list.discard(local_tuple)
                    del x.cluster2tuple[v]
                except KeyError:
                    pass

                self.links[x.points[0], w.points[0]] = self.links[x.points[0], u.points[0]] + self.links[
                    x.points[0], v.points[0]]
                g_measure = self.goodness_measure(w, x)

                tup = (g_measure, w)
                x.sorted_list.add(tup)
                x.cluster2tuple[w] = tup

                tup = (g_measure, x)
                w.sorted_list.add(tup)
                w.cluster2tuple[x] = tup
                self.global_sorted_list.add(x)

            self.global_sorted_list.add(w)
            print("Clusters left: {}".format(len(self.global_sorted_list)))

    def compute_links(self) -> np.array:
        """Returns a matrix of links, O(n*m_a^2) complexity
        where m_a is an average number of neighbors for each cluster-point"""

        neighbors_list = self.find_neighbors()
        links = np.zeros((self.S.shape[0], self.S.shape[0]), dtype=int)
        n_rows, n_col = self.S.shape
        for i in range(0, n_rows):
            i_neighbors = neighbors_list[i][0]
            for j in range(0, i_neighbors.shape[0] - 1):
                for l in range(j + 1, i_neighbors.shape[0]):
                    links[i_neighbors[j], i_neighbors[l]] = links[i_neighbors[j], i_neighbors[l]] + 1
                    links[i_neighbors[l], i_neighbors[j]] = links[i_neighbors[l], i_neighbors[j]] + 1
            # print("Links calcuated for {} points".format(i))
        return links

    def find_neighbors(self) -> list:
        n_rows, n_col = self.S.shape
        neighbors_list = [None for i in range(0, n_rows)]
        for i in range(0, n_rows):
            feature_similarities_and = np.logical_and(self.S, self.S[i, :])
            feature_similarities_and[i, :] = False  # So that point isn't his own neighbor
            feature_similarities_or = np.logical_or(self.S, self.S[i, :])
            similarity = tanimoto_coefficient(feature_similarities_and, feature_similarities_or)
            neighbors_list[i] = np.where(similarity >= self.theta)
        return neighbors_list

    def goodness_measure(self, c1: Cluster, c2: Cluster) -> float:
        numerator = 0
        for i in c1.points:
            for j in c2.points:
                numerator += self.links[i, j]
        denominator = (len(c1.points) + len(c1.points)) ** self.goodness_exponent - len(
            c1.points) ** self.goodness_exponent - len(
            c2.points) ** self.goodness_exponent
        return (-1) * numerator / denominator

    def clusters(self) -> list:
        return [x for x in self.q if x is not None]


class RockGeoClustering(RockClustering):
    """ This is adaptation of rock clustering to geographical data """

    def __init__(self, S: np.array, k: int, theta=0.5, nbr_max_distance=100):
        """
        :param S: Data set to cluster
        :param k: Number of clusters
        :param theta: Theta parameter from the original paper, should be from 0 to 1
        :param nbr_max_distance: Distance in meters below which to classify points as neighbors
        """
        self.nbr_max_distance = nbr_max_distance
        super().__init__(S, k, theta)

    def find_neighbors(self) -> list:
        n_rows, n_col = self.S.shape
        neighbors_list = [None for i in range(0, n_rows)]
        for i in range(0, n_rows):
            distance_vector = spherical_distance(self.S[i, 0], self.S[i, 1], self.S[:, 0], self.S[:, 1])
            distance_vector[i] = np.inf  # So that point isn't his own neighbor
            neighbors_list[i] = np.where(distance_vector <= self.nbr_max_distance)
        return neighbors_list


class RockRealClustering(RockClustering):
    """ This is adaptation of rock clustering to real data """

    def __init__(self, S: np.array, k: int, theta=0.5, nbr_max_distance=1):
        """
        :param S: Data set to cluster
        :param k: Number of clusters
        :param theta: Theta parameter from the original paper, should be from 0 to 1
        :param nbr_max_distance: Distance in meters below which to classify points as neighbors
        """
        self.nbr_max_distance = nbr_max_distance
        super().__init__(S, k, theta)

    def find_neighbors(self) -> list:
        n_rows, n_col = self.S.shape
        neighbors_list = [None for i in range(0, n_rows)]
        for i in range(0, n_rows):
            distance_vector = euclidean_distance(self.S[i, :], self.S)
            distance_vector[i] = np.inf  # So that point isn't his own neighbor
            neighbors_list[i] = np.where(distance_vector <= self.nbr_max_distance)
        return neighbors_list

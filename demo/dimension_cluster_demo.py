import numpy as np

# dataset sample
from sklearn.datasets import make_blobs
from mpl_toolkits.mplot3d import Axes3D

# kmeans++
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
# kmeans
from pyclustering.cluster.kmeans import kmeans
# custom distance metric
from pyclustering.utils.metric import type_metric, distance_metric
# visualizer
import matplotlib.pyplot as plt
from pyclustering.cluster import cluster_visualizer

def kmeans_plus_plus_demo(data):
    # number of centroid
    amount_centers = 4

    # candidates options
    # actually i don' know yet
    amount_candidates = kmeans_plusplus_initializer.FARTHEST_CENTER_CANDIDATE
    centroid_initializer = kmeans_plusplus_initializer(data=X,
                                          amount_centers=amount_centers,
                                          amount_candidates=amount_candidates)
    initialized_center = centroid_initializer.initialize()

    # visualize initialized centroid
    visualizer = cluster_visualizer()
    visualizer.append_cluster(data)
    visualizer.append_cluster(initialized_center, marker='*', markersize=100)
    visualizer.show()

    return initialized_center

def kmeans_demo(data, initialized_centers):

    def l2norm(point1, point2):
        # custom distance metric: L2 norm
        # it only get 2 parameters that point1, point2
        # point is mean that specific point coordinates
        # if point is 2d-space that will be represent [a, b] as ndarray
        # if point is 3d-space that will be represent [a, b, c] as ndarray

        return np.sum(np.sqrt(np.square(point1 - point2)))

    metric = distance_metric(type_metric.USER_DEFINED, func=l2norm)

    # create K-Means algorithm with specific distance metric
    kmeans_obj = kmeans(data=data,
                        initial_centers=initialized_centers,
                        metric=metric)

    kmeans_obj.process()
    clusters = kmeans_obj.get_clusters()
    centers = kmeans_obj.get_centers()

    visualizer = cluster_visualizer()
    visualizer.append_clusters(clusters=clusters, data=data)
    visualizer.append_cluster(centers, marker='*', markersize=100)
    visualizer.show()

    return centers


if __name__ == "__main__":

    # load dataset & visualize
    X, y = make_blobs(n_samples=800, n_features=3, centers=4)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2])
    plt.show(fig)
    plt.close()

    initialized_centers = kmeans_plus_plus_demo(data=X)
    print(initialized_centers)
    result_centers = kmeans_demo(data=X,
                           initialized_centers = initialized_centers)
    print(result_centers)

import math

# dataset sample
from sklearn.datasets import make_blobs
from mpl_toolkits.mplot3d import Axes3D

# kmeans++
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
# kmeans
from pyclustering.cluster.kmeans import kmeans
# custom distance metric
from pyclustering.utils.metric import type_metric
# visualizer
import matplotlib.pyplot as plt
from pyclustering.cluster import cluster_visualizer
from pyclustering.samples.definitions import IMAGE_SIMPLE_SAMPLES
from pyclustering.utils import draw_image_mask_segments

import numpy as np

class distance_metric:
    """!
    @brief Distance metric performs distance calculation between two points in line with encapsulated function, for
            example, euclidean distance or chebyshev distance, or even user-defined.
    @details
    Example of Euclidean distance metric:
    @code
        metric = distance_metric(type_metric.EUCLIDEAN)
        distance = metric([1.0, 2.5], [-1.2, 3.4])
    @endcode
    Example of Chebyshev distance metric:
    @code
        metric = distance_metric(type_metric.CHEBYSHEV)
        distance = metric([0.0, 0.0], [2.5, 6.0])
    @endcode
    In following example additional argument should be specified (generally, 'degree' is a optional argument that is
     equal to '2' by default) that is specific for Minkowski distance:
    @code
        metric = distance_metric(type_metric.MINKOWSKI, degree=4)
        distance = metric([4.0, 9.2, 1.0], [3.4, 2.5, 6.2])
    @endcode
    User may define its own function for distance calculation:
    @code
        user_function = lambda point1, point2: point1[0] + point2[0] + 2
        metric = distance_metric(type_metric.USER_DEFINED, func=user_function)
        distance = metric([2.0, 3.0], [1.0, 3.0])
    @endcode
    """
    def __init__(self, metric_type, **kwargs):
        """!
        @brief Creates distance metric instance for calculation distance between two points.
        @param[in] metric_type (type_metric):
        @param[in] **kwargs: Arbitrary keyword arguments (available arguments: 'numpy_usage' 'func' and corresponding additional argument for
                    for specific metric types).
        <b>Keyword Args:</b><br>
            - func (callable): Callable object with two arguments (point #1 and point #2) or (object #1 and object #2) in case of numpy usage.
                                This argument is used only if metric is 'type_metric.USER_DEFINED'.
            - degree (numeric): Only for 'type_metric.MINKOWSKI' - degree of Minkowski equation.
            - numpy_usage (bool): If True then numpy is used for calculation (by default is False).
        """
        self.__type = metric_type
        self.__args = kwargs
        self.__func = self.__args.get('func', None)
        self.__numpy = self.__args.get('numpy_usage', False)

        self.__calculator = self.__create_distance_calculator()


    def __call__(self, point1, point2):
        """!
        @brief Calculates distance between two points.
        @param[in] point1 (list): The first point.
        @param[in] point2 (list): The second point.
        @return (double) Distance between two points.
        """
        return self.__calculator(point1, point2)


    def get_type(self):
        """!
        @brief Return type of distance metric that is used.
        @return (type_metric) Type of distance metric.
        """
        return self.__type


    def get_arguments(self):
        """!
        @brief Return additional arguments that are used by distance metric.
        @return (dict) Additional arguments.
        """
        return self.__args


    def get_function(self):
        """!
        @brief Return user-defined function for calculation distance metric.
        @return (callable): User-defined distance metric function.
        """
        return self.__func


    def enable_numpy_usage(self):
        """!
        @brief Start numpy for distance calculation.
        @details Useful in case matrices to increase performance. No effect in case of type_metric.USER_DEFINED type.
        """
        self.__numpy = True
        if self.__type != type_metric.USER_DEFINED:
            self.__calculator = self.__create_distance_calculator()


    def disable_numpy_usage(self):
        """!
        @brief Stop using numpy for distance calculation.
        @details Useful in case of big amount of small data portion when numpy call is longer than calculation itself.
                  No effect in case of type_metric.USER_DEFINED type.
        """
        self.__numpy = False
        self.__calculator = self.__create_distance_calculator()


    def __create_distance_calculator(self):
        """!
        @brief Creates distance metric calculator.
        @return (callable) Callable object of distance metric calculator.
        """
        if self.__numpy is True:
            return self.__create_distance_calculator_numpy()

        return self.__create_distance_calculator_basic()


    def __create_distance_calculator_basic(self):
        """!
        @brief Creates distance metric calculator that does not use numpy.
        @return (callable) Callable object of distance metric calculator.
        """
        if self.__type == type_metric.USER_DEFINED:
            return self.__func

        else:
            raise ValueError("Unknown type of metric: '%d'", self.__type)


    def __create_distance_calculator_numpy(self):
        """!
        @brief Creates distance metric calculator that uses numpy.
        @return (callable) Callable object of distance metric calculator.
        """
        if self.__type == type_metric.USER_DEFINED:
            return self.__func

        else:
            raise ValueError("Unknown type of metric: '%d'", self.__type)


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

    # custom distance metric: L2 norm

    l2_norm = lambda point1, point2, point3: \
        math.sqrt(sum([pow(point1, 2), pow(point2, 2), pow(point3, 2)]))
    metric = distance_metric(type_metric.USER_DEFINED, func=l2_norm)


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
    centers = kmeans_demo(data=X,
                           initialized_centers = initialized_centers)
    print(centers)



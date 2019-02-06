from utils.dimension_cluster import DimensionCluster

root = "/home/martin/Documents/dev/Deepbaksu_vision/Datasets/VOCdevkit/VOC2007"

# distance metric can select "iou" or "l2"
dim = DimensionCluster(root=root, display=True, distance_metric="l2")
prior_boxes = dim.process()
print(prior_boxes)
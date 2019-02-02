
import numpy as np
import imgaug as ia
from PIL import Image
from mpl_toolkits.axes_grid1 import ImageGrid


def GetYoloStyleBBoxes(normed_lxywhs, bbs_aug, image_width, image_height):
    normed_bbs_aug = []

    for i in range(len(bbs_aug.bounding_boxes)):
        after = bbs_aug.bounding_boxes[i]
        coord = CvtCoordsXXYY2XYWH(image_width, image_height, xmin=after.x1, xmax=after.x2, ymin=after.y1, ymax=after.y2)
        normed_bbs_aug.append([normed_lxywhs[i][0], coord[0], coord[1], coord[2], coord[3]])

    return normed_bbs_aug

def GetImgaugStyleBBoxes(normed_lxywhs, image_width, image_height):
    bbs = ia.BoundingBoxesOnImage([], shape=(image_width, image_height))

    for normed_lxywh in normed_lxywhs:
        xxyy = CvtCoordsXYWH2XXYY(normed_lxywh, image_width, image_height)
        bbs.bounding_boxes.append(ia.BoundingBox(x1=xxyy[0], x2=xxyy[1], y1=xxyy[2], y2=xxyy[3], label='None'))

    return bbs

def CvtCoordsXXYY2XYWH(image_width, image_height, xmin, xmax, ymin, ymax):
    # calculate bbox_center
    bbox_center_x = (xmin + xmax) / 2
    bbox_center_y = (ymin + ymax) / 2

    # calculate bbox_size
    bbox_width = xmax - xmin
    bbox_height = ymax - ymin

    # normalize
    normalized_x = bbox_center_x / image_width
    normalized_y = bbox_center_y / image_height
    normalized_w = bbox_width / image_width
    normalized_h = bbox_height / image_height

    return normalized_x, normalized_y, normalized_w, normalized_h

def CvtCoordsXYWH2XXYY(normed_lxywh, image_width, image_height):
    centered_x = normed_lxywh[1] * image_width
    centered_y = normed_lxywh[2] * image_height
    object_width = normed_lxywh[3] * image_width
    object_height = normed_lxywh[4] * image_height

    xmin = centered_x - object_width / 2
    xmax = centered_x + object_width / 2
    ymin = centered_y - object_height / 2
    ymax = centered_y + object_height / 2

    return xmin, xmax, ymin, ymax


class Augmenter(object):

    def __init__(self, seq):
        self.seq = seq

    def __call__(self, pil_image, box_annotation_dict):

        #[3.0, 0.498, 0.508, 0.328, 0.661]
        #(149.63199999999998, 296.576, 79.51999999999998, 375.648)

        image = np.array(pil_image)

        image_aug, normed_bbs_aug = self.augmentImage(image, box_annotation_dict, self.seq)

        image_aug = Image.fromarray(image_aug)  # numpy array to PIL image Again!
        return image_aug, normed_bbs_aug

    @staticmethod
    def augmentImage(image, box_annotation_dict, seq):

        bbs = GetImgaugStyleBBoxes(normed_lxywhs, image_width, image_height)

        seq_det = seq.to_deterministic()

        image_aug = seq_det.augment_images([image])[0]

        bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]

        bbs_aug = bbs_aug.remove_out_of_image().cut_out_of_image()

        normed_bbs_aug = GetYoloStyleBBoxes(normed_lxywhs, bbs_aug, image_width, image_height)

        return image_aug, normed_bbs_aug

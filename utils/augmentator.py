import numpy as np
import imgaug as ia
from PIL import Image

class Augmenter():

    def __init__(self, seq):
        self.seq = seq

    def __call__(self, img_and_annotation):

        image = img_and_annotation[0]
        box_annotation_dict = img_and_annotation[1]

        image = np.array(image)
        image_aug, bbs_aug = self.augment_image(image, box_annotation_dict, self.seq)
        image_aug = Image.fromarray(image_aug)

        return image_aug, bbs_aug

    def augment_image(self, image, box_annotation_dict, seq):
        bbs = self.transform_imgaug_style_boxes(box_annotation_dict)
        seq_det = seq.to_deterministic()

        image_aug = seq_det.augment_images([image])[0]
        bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]
        bbs_aug = bbs_aug.remove_out_of_image().cut_out_of_image()
        augmented_box = self.transofrm_annotation_information_style(box_annotation_dict, bbs_aug)

        return image_aug, augmented_box

    @staticmethod
    def transofrm_annotation_information_style(box_annotation_dict, bbs_aug):
        assert isinstance(box_annotation_dict, dict)

        box_annotation_keys = box_annotation_dict.keys()
        assert "size" in box_annotation_keys
        assert "object" in box_annotation_keys

        size_tag_keys = box_annotation_dict["size"].keys()
        assert "width" in size_tag_keys
        assert "height" in size_tag_keys
        assert "depth" in size_tag_keys

        assert isinstance(box_annotation_dict["object"], list)
        for _object in box_annotation_dict["object"]:
            _object_keys = _object.keys()
            assert "name" in _object_keys
            assert "xmin" in _object_keys
            assert "ymin" in _object_keys
            assert "xmax" in _object_keys
            assert "ymax" in _object_keys

        assert isinstance(bbs_aug, ia.BoundingBoxesOnImage)

        objects = box_annotation_dict["object"]
        objects.clear()

        for i in range(len(bbs_aug.bounding_boxes)):
            augmented_box = bbs_aug.bounding_boxes[i]
            objects.append(
                {
                    "name": augmented_box.label,
                    "xmin": augmented_box.x1,
                    "ymin": augmented_box.y1,
                    "xmax": augmented_box.x2,
                    "ymax": augmented_box.y2
                }
            )

        return box_annotation_dict

    @staticmethod
    def transform_imgaug_style_boxes(box_annotation_dict):
        assert isinstance(box_annotation_dict, dict)

        box_annotation_keys = box_annotation_dict.keys()
        assert "size" in box_annotation_keys
        assert "object" in box_annotation_keys

        size_tag_keys = box_annotation_dict["size"].keys()
        assert "width" in size_tag_keys
        assert "height" in size_tag_keys
        assert "depth" in size_tag_keys

        assert isinstance(box_annotation_dict["object"], list)
        for _object in box_annotation_dict["object"]:
            _object_keys = _object.keys()
            assert "name" in _object_keys
            assert "xmin" in _object_keys
            assert "ymin" in _object_keys
            assert "xmax" in _object_keys
            assert "ymax" in _object_keys

        image_width = int(box_annotation_dict["size"]["width"])
        image_height = int(box_annotation_dict["size"]["height"])

        bbs = ia.BoundingBoxesOnImage([], shape=(image_height, image_width))

        for _object in box_annotation_dict["object"]:
            name = _object["name"]
            xmin = int(_object["xmin"])
            ymin = int(_object["ymin"])
            xmax = int(_object["xmax"])
            ymax = int(_object["ymax"])
            bbs.bounding_boxes.append(ia.BoundingBox(x1=xmin,
                                                     x2=xmax,
                                                     y1=ymin,
                                                     y2=ymax,
                                                     label=name))

        return bbs

import os
import numpy as np
from torch.utils import data

class VocDetection(data.Dataset):
    """ `PASCAL VOC Detection <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Dataset.

    Args:
        root (string): Root directory where images and annotations were downloaded
        imageFolder (string) : Path to images folder. image have extension as jpg, jpeg, png
        annotationFolder (string) : Path to annotation file folder. annotation file have extension as xml
        transform (callable, optional): A function/transform that  takes in an PIL image and label
            and returns a transformed version. E.g, ``transforms.ToTensor``
    """

    def __init__(self, root, image_folder="JPEGImages", annotation_folder="Annotations", transform=None):

        self.root = root
        self.image_folder = image_folder
        self.annotation_folder = annotation_folder

        self.image_paths = self._get_filepaths(self.image_folder)
        self.image_paths.sort(key=lambda filepath : int(filepath.split("/")[-1].split(".")[0]))

        self.annotation_paths = self._get_filepaths(self.annotation_folder)
        self.annotation_paths.sort(key=lambda filepath : int(filepath.split("/")[-1].split(".")[0]))

        self.classes = self._get_classes_list(self.annotation_paths)

        self._validate_image_Extension(self.image_paths)
        assert(self.__is_length_same(image_paths = self.image_paths,
                                     annotation_paths = self.annotation_paths))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``method``.
        """
        # TODO it shoould be Implement

        pass

    def _get_classes_list(self, annotation_paths):
        import xml.etree.ElementTree as Et
        # [x] TODO write docstring
        # [v] TODO add assertion
        # [v] TODO getting classes list.
        # [v] TODO search annotation label
        # [v] TODO check working well

        # result sorted ordered by alphabet
        assert(len(annotation_paths) != 0)

        classes = list()
        for annotation_path in annotation_paths:
            xml_file = open(annotation_path, "r")
            tree = Et.parse(xml_file)
            root = tree.getroot()

            objects = root.findall("object")
            assert(len(objects) != 0)

            for _object in objects:
                cls_tag = _object.find("name")
                assert(cls_tag != None)
                cls = cls_tag.text
                classes.append(cls) if cls not in classes else None
                classes = list(filter(None, classes))

        classes.sort()
        return classes

    def _get_label(self):
        # TODO write docstring
        # TODO getting detection label
        pass

    def __convert_to_yolo_label(self):
        # TODO write docstring
        # TODO convert to yolo label
        pass

    def __parse_xml_of_voc(self, annotation_path):
        import xml.etree.ElementTree as Et
        # [x] TODO write docstring
        # [x] TODO it should check about satisfied it have all element for detection
        # [v] TODO getting voc annotation
        # [v] TODO check working well

        result = dict()

        xml_file = open(annotation_path, "r")
        tree = Et.parse(xml_file)
        root = tree.getroot()

        size_tag = root.find("size")

        result["size"] = {
            "width": size_tag.find("width").text,
            "height": size_tag.find("height").text,
            "depth": size_tag.find("depth").text
        }

        result["object"] = list()

        objects = root.findall("object")
        assert (len(objects) != 0)

        for _object in objects:

            name_tag = _object.find("name")
            bndbox_tag = _object.find("bndbox")
            assert (name_tag != None)
            assert (bndbox_tag != None)

            xmin_tag = bndbox_tag.find("xmin")
            ymin_tag = bndbox_tag.find("ymin")
            xmax_tag = bndbox_tag.find("xmax")
            ymax_tag = bndbox_tag.find("ymax")

            assert (xmin_tag != None)
            assert (ymin_tag != None)
            assert (xmax_tag != None)
            assert (ymax_tag != None)

            result["object"].append({
                "name": name_tag.text,
                "xmin": xmin_tag.text,
                "ymin": ymin_tag.text,
                "xmax": xmax_tag.text,
                "ymax": ymax_tag.text
            })

        return result

    def _get_filepaths(self, subdirectory):
        (dir_path, _, filenames) = next(os.walk(os.path.abspath(os.path.join(self.root, subdirectory))))
        return [os.path.join(dir_path, filename) for filename in filenames]


    def _validate_image_Extension(self, image_paths):
        # [x] TODO add assertion
        validations_jpg = self.__validate_extension(image_paths, "jpg")
        validations_jpeg = self.__validate_extension(image_paths, "jpeg")
        validations_png = self.__validate_extension(image_paths, "png")


        integration_validation = self.__combine_validations(jpg_validations = validations_jpg,
                                                            jpeg_validations = validations_jpeg,
                                                            png_validations = validations_png)

        assert(self.__combine_validation_elements(integration_validation))


    def __combine_validation_elements(self, validations):
        # [x] TODO add assertion
        intergated_validation = True

        for validation in validations:
            intergated_validation = intergated_validation and validation

        return intergated_validation


    def __combine_validations(self, **kwargs):
        # [x] TODO add assertion
        validations = list()
        for key in kwargs.keys():
            validations.append(kwargs[key])

        result = validations[-1]

        for i in range(len(validations) - 1):
            result = [result[j] or validations[i][j] for j in range(len(result))]
        return result


    def __is_length_same(self, **kwargs):
        # [x] TODO add assertion
        lengths = list()
        comparisions = list()
        flag = True
        for key in kwargs.keys():
            lengths.append(len(kwargs[key]))

        for i in range(len(lengths)-1):
            comparisions.append((lengths[-1] == lengths[i]))

        for comparision in comparisions:
            flag = flag and comparision

        return flag


    def __validate_extension(self, filenames, extension):
        # [x] TODO add assertion
        return [filename[-3:].lower() == extension for filename in filenames]

root = "/home/martin/Documents/dev/Deepbaksu_vision/_Datasets/VOC2012/"
voc = VocDetection(root=root)
print(voc.annotation_paths[0])
print(voc._parse_xml_of_voc(voc.annotation_paths[0]))
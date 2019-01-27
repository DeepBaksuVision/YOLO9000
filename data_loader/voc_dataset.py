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
        self.annotations_paths = self._get_filepaths(self.annotation_folder)
        self.annotations_paths.sort(key=lambda filepath : int(filepath.split("/")[-1].split(".")[0]))

        assert(self._validate_image_Extension())
        assert(self.__is_length_same(image_paths = self.image_paths,
                                     annotation_paths = self.annotations_paths))


    def _get_filepaths(self, subdirectory):
        (dir_path, _, filenames) = next(os.walk(os.path.abspath(os.path.join(self.root, subdirectory))))
        return [os.path.join(dir_path, filename) for filename in filenames]


    def _validate_image_Extension(self):
        validations_jpg = self.__validate_extension(self.image_paths, "jpg")
        validations_jpeg = self.__validate_extension(self.image_paths, "jpeg")
        validations_png = self.__validate_extension(self.image_paths, "png")


        integration_validation = self.__compare_validations(jpg_validations = validations_jpg,
                                                            jpeg_validations = validations_jpeg,
                                                            png_validations = validations_png)

        return self.__compare_validation_elements(integration_validation)


    def __compare_validation_elements(self, validations):
        intergated_validation = True

        for validation in validations:
            intergated_validation = intergated_validation and validation

        return intergated_validation


    def __compare_validations(self, **kwargs):

        validations = list()
        for key in kwargs.keys():
            validations.append(kwargs[key])

        result = validations[-1]

        for i in range(len(validations) - 1):
            result = [result[j] or validations[i][j] for j in range(len(result))]
        return result


    def __is_length_same(self, **kwargs):
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
        return [filename[-3:].lower() == extension for filename in filenames]

root = "/home/martin/Documents/dev/Deepbaksu_vision/Datasets/VOCdevkit/VOC2007/"
voc = VocDetection(root=root)
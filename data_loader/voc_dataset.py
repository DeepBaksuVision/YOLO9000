import os
from PIL import Image
from torch.utils import data

class VocDetection(data.Dataset):
    """ `PASCAL VOC Detection <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Dataset.

    Args:
        root (string):
            Root directory where images and annotations were downloaded
        imageFolder (string) :
            Path to images folder. image have extension as jpg, jpeg, png
        annotationFolder (string) :
            Path to annotation file folder. annotation file have extension as xml
        transform (callable, optional): A function/transform that  takes in an PIL image and label
            and returns a transformed version. E.g, ``transforms.ToTensor``
    """

    def __init__(self,
                 root,
                 image_folder="JPEGImages",
                 annotation_folder="Annotations",
                 resize_size=(416, 416),
                 transform=None):

        self.resize_size = resize_size

        image_path_list = self._get_list_of_filepath(root, image_folder)
        self.image_path_list = self.sort_ordered_by_filename(image_path_list)

        annotation_path_list = self._get_list_of_filepath(root, annotation_folder)
        self.annotation_path_list = self.sort_ordered_by_filename(annotation_path_list)

        classes_list = self._get_classes_list_from_dataset(self.annotation_path_list)
        classes_list_ordered_by_alphabet = sorted(classes_list)
        self.classes_list = classes_list_ordered_by_alphabet

        self.transform = transform

        assert self._is_validated_image_Extension(self.image_path_list)
        assert self._is_validated_annotation_Extension(self.annotation_path_list)
        assert len(self.image_path_list) == len(self.annotation_path_list)

    @staticmethod
    def sort_ordered_by_filename(path_list):
        assert isinstance(path_list, list)
        for path in path_list:
            assert isinstance(path, str)

        return sorted(path_list,
                   key=lambda filepath: int(filepath.split("/")[-1].split(".")[0]))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``method``.
        """
        assert isinstance(index, int)

        image = Image.open(self.image_path_list[index]).convert('RGB')
        box_annotation_dict = self.__parse_voc(self.annotation_path_list[index])

        if self.transform:
            image, box_annotation_dict = self.transform([image, box_annotation_dict])

        target = self.__convert_box_label_to_yolo_label(box_annotation_dict, self.classes_list)
        image = image.resize(self.resize_size)

        return image, target


    def __len__(self):
        return len(self.image_path_list)

    @staticmethod
    def _get_classes_list_from_dataset(annotation_path_list):
        import xml.etree.ElementTree as Et

        assert isinstance(annotation_path_list, list)
        for annotation_path in annotation_path_list:
            assert isinstance(annotation_path, str)
        assert annotation_path_list

        classes_list = list()
        for annotation_path in annotation_path_list:
            xml_file = open(annotation_path, "r")
            tree = Et.parse(xml_file)

            element_list = list()
            for elem in tree.iter():
                element_list.append(elem.tag)

            assert "name" in element_list

            root = tree.getroot()

            objects = root.findall("object")
            assert objects

            for _object in objects:
                cls_tag = _object.find("name")
                cls = cls_tag.text
                if cls not in classes_list:
                    classes_list.append(cls)

                classes_list = list(filter(None, classes_list))

        return classes_list

    def __convert_box_label_to_yolo_label(self, label, classes_list):
        assert isinstance(label, dict)
        assert isinstance(classes_list, list)
        for cls in classes_list:
            assert isinstance(cls, str)

        root_keys = label.keys()
        size_keys = label["size"].keys()
        number_of_objects = len(label["object"])

        assert "size" in root_keys
        assert "object" in root_keys
        assert "width" in size_keys
        assert "height" in size_keys
        assert number_of_objects != 0

        yolo_label = list()

        image_size = {
            "width": float(label["size"]["width"]),
            "height": float(label["size"]["height"]),
        }

        for _object in label["object"]:
            _object_keys = _object.keys()
            assert "name" in _object_keys
            assert "xmin" in _object_keys
            assert "ymin" in _object_keys
            assert "xmax" in _object_keys
            assert "ymax" in _object_keys

            name = _object["name"]
            cls = float(classes_list.index(name))
            box_coordinate = {
                "xmin": float(_object["xmin"]),
                "ymin": float(_object["ymin"]),
                "xmax": float(_object["xmax"]),
                "ymax": float(_object["ymax"]),
            }

            yolo_coordinate = self.__convert_coordinate(image_size, box_coordinate)
            yolo_coordinate.insert(0, cls)
            yolo_label.append(yolo_coordinate)

        return yolo_label

    @staticmethod
    def __convert_coordinate(image_size, box_coordinate):
        image_size_keys = image_size.keys()
        box_coordinate_keys = box_coordinate.keys()
        assert "width" in image_size_keys
        assert "height" in image_size_keys
        assert "xmin" in box_coordinate_keys
        assert "ymin" in box_coordinate_keys
        assert "xmax" in box_coordinate_keys
        assert "ymax" in box_coordinate_keys
        assert isinstance(image_size, dict)
        assert isinstance(box_coordinate, dict)
        assert isinstance(image_size["width"], float)
        assert isinstance(image_size["height"], float)
        assert isinstance(box_coordinate["xmin"], float)
        assert isinstance(box_coordinate["ymin"], float)
        assert isinstance(box_coordinate["xmax"], float)
        assert isinstance(box_coordinate["ymax"], float)

        x_of_box = (box_coordinate["xmin"] + box_coordinate["xmax"]) / 2.0
        y_of_box = (box_coordinate["ymin"] + box_coordinate["ymax"]) / 2.0
        width_of_box = box_coordinate["xmax"] - box_coordinate["xmin"]
        height_of_box = box_coordinate["ymax"] - box_coordinate["ymin"]

        relative_x_of_center = x_of_box / image_size["width"]
        relative_y_of_center = y_of_box / image_size["height"]
        relative_box_width = width_of_box / image_size["width"]
        relative_box_height = height_of_box / image_size["height"]

        return [relative_x_of_center, relative_y_of_center,
                relative_box_width, relative_box_height]

    @staticmethod
    def __parse_voc(annotation_path):
        import xml.etree.ElementTree as Et
        assert isinstance(annotation_path, str)

        xml_file = open(annotation_path, "r")
        tree = Et.parse(xml_file)

        element_list = list()
        for elem in tree.iter():
            element_list.append(elem.tag)

        assert "size" in element_list
        assert "width" in element_list
        assert "height" in element_list
        assert "object" in element_list
        assert "name" in element_list
        assert "bndbox" in element_list
        assert "xmin" in element_list
        assert "ymin" in element_list
        assert "xmax" in element_list
        assert "ymax" in element_list

        result = dict()
        root = tree.getroot()

        size_tag = root.find("size")

        result["size"] = {
            "width": size_tag.find("width").text,
            "height": size_tag.find("height").text,
            "depth": size_tag.find("depth").text
        }

        result["object"] = list()

        objects = root.findall("object")
        assert objects

        for _object in objects:

            result["object"].append({
                "name": _object.find("name").text,
                "xmin": _object.find("bndbox").find("xmin").text,
                "ymin": _object.find("bndbox").find("ymin").text,
                "xmax": _object.find("bndbox").find("xmax").text,
                "ymax": _object.find("bndbox").find("ymax").text
            })

        return result


    @staticmethod
    def _get_list_of_filepath(root_directory, sub_directory):
        assert isinstance(root_directory, str)
        assert isinstance(sub_directory, str)

        (dir_path, _, filenames) = \
            next(os.walk(os.path.abspath(os.path.join(root_directory, sub_directory))))
        return [os.path.join(dir_path, filename) for filename in filenames]


    def _is_validated_annotation_Extension(self, annotation_path_list):
        assert isinstance(annotation_path_list, list)
        for path in annotation_path_list:
            assert isinstance(path, str)

        validation_list_about_xml = self.__is_validated_extension(annotation_path_list, "xml")

        return self.__is_all_element_validated(validation_list_about_xml)


    def _is_validated_image_Extension(self, image_path_list):
        assert isinstance(image_path_list, list)
        for path in image_path_list:
            assert isinstance(path, str)

        validation_list_about_jpg = self.__is_validated_extension(image_path_list, "jpg")
        validation_list_about_jpeg = self.__is_validated_extension(image_path_list, "jpeg")
        validation_list_about_png = self.__is_validated_extension(image_path_list, "png")


        integrated_validation_list = \
            self.__is_supported_image_extension(jpg_validations=validation_list_about_jpg,
                                                jpeg_validations=validation_list_about_jpeg,
                                                png_validations=validation_list_about_png)

        return self.__is_all_element_validated(integrated_validation_list)


    @staticmethod
    def __is_all_element_validated(validation_list):
        assert isinstance(validation_list, list)
        for validation_value in validation_list:
            assert isinstance(validation_value, bool)

        intergated_validation = True

        for validation in validation_list:
            intergated_validation = intergated_validation and validation

        return intergated_validation


    @staticmethod
    def __is_supported_image_extension(**kwargs):
        keys = kwargs.keys()
        assert "jpg_validations" in keys
        assert "jpeg_validations" in keys
        assert "png_validations" in keys

        validations = list()
        for key in keys:
            validations.append(kwargs[key])

        # Integrate all validation
        # initial list was selected last of validation list
        # because of last of validation list not selected for loop
        result = validations[-1]

        for i in range(len(validations) - 1):
            result = [result[j] or validations[i][j] for j in range(len(result))]

        return result


    @staticmethod
    def __is_validated_extension(file_list, extension):
        assert isinstance(file_list, list)
        assert isinstance(extension, str)

        return [filename[-3:].lower() == extension for filename in file_list]

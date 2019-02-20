import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

def visualize_detection_label(image, coordinates, classes_list, grid_size, mode="yolo"):
    assert image
    assert coordinates
    assert classes_list
    assert isinstance(classes_list, list)
    assert isinstance(grid_size, tuple)
    assert len(grid_size) == 2
    assert isinstance(image, Image.Image)

    draw = ImageDraw.Draw(image)
    draw_grid(draw, image.size, grid_size)

    if mode == "yolo":
        assert isinstance(coordinates, list)
        for coordinate in coordinates:
            assert len(coordinate) == 5
            for element in coordinate:
                assert isinstance(element, float)

        for coordinate in coordinates:
            draw_yolo_information(draw, coordinate, image.size, classes_list)
    elif mode == "box":
        draw_box_information(draw, coordinates)


    plt.figure()
    plt.imshow(image)
    plt.show()
    plt.close()

def draw_grid(draw_image_obj, image_size, grid_size):
    assert isinstance(draw_image_obj, ImageDraw.ImageDraw)
    assert isinstance(image_size, tuple)
    assert isinstance(grid_size, tuple)
    assert len(image_size) == 2
    assert len(grid_size) == 2

    image_width, image_height = image_size
    grid_width, grid_height = grid_size

    dx = image_width // grid_width
    dy = image_height // grid_height

    for i in range(0, image_width, dx):
        y_axis_line = ((i, 0), (i, image_height))
        draw_image_obj.line(y_axis_line, fill="red")

    for i in range(0, image_height, dy):
        x_axis_line = ((0, i), (image_width, i))
        draw_image_obj.line(x_axis_line, fill="red")

def convert_yolo_label_to_box_label(coordinate, image_size, classes_list):
    assert isinstance(coordinate, list)
    assert len(coordinate) == 5
    for element in coordinate:
        assert isinstance(element, float)
    assert isinstance(image_size, tuple)
    assert isinstance(classes_list, list)
    for cls in classes_list:
        isinstance(cls, str)

    image_width, image_height = image_size

    cls = classes_list[int(coordinate[0])]
    x_of_center = int(coordinate[1] * image_width)
    y_of_center = int(coordinate[2] * image_height)
    width = coordinate[3] * image_width
    height = coordinate[4] * image_height

    xmin = int(x_of_center - (width / 2))
    ymin = int(y_of_center - (height / 2))
    xmax = int(xmin + width)
    ymax = int(ymin + height)

    return [cls, xmin, ymin, xmax, ymax]

def draw_box_information(draw_image_obj, coordinates):

    for _object in coordinates["object"]:
        cls = _object["name"]
        xmin = int(_object["xmin"])
        ymin = int(_object["ymin"])
        xmax = int(_object["xmax"])
        ymax = int(_object["ymax"])
        draw_image_obj.rectangle(((xmin, ymin), (xmax, ymax)), outline="blue")
        draw_image_obj.text((xmin, ymin), cls)

def draw_yolo_information(draw_image_obj, coordinate, image_size, classes_list):
    assert isinstance(draw_image_obj, ImageDraw.ImageDraw)
    assert isinstance(image_size, tuple)
    assert isinstance(coordinate, list)
    assert len(coordinate) == 5
    for element in coordinate:
        assert isinstance(element, float)

    image_width, image_height = image_size

    x_of_center = int(coordinate[1] * image_width)
    y_of_center = int(coordinate[2] * image_height)

    cls, xmin, ymin, xmax, ymax = \
        convert_yolo_label_to_box_label(coordinate, image_size, classes_list)

    draw_image_obj.rectangle(((xmin, ymin), (xmax, ymax)), outline="blue")
    draw_image_obj.ellipse(((x_of_center - 2, y_of_center - 2),
                  (x_of_center + 2, y_of_center + 2)),
                 fill='blue')
    draw_image_obj.text((xmin, ymin), cls)

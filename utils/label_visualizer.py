import numpy as np
import torchvision.transforms as transforms
from PIL import Image, ImageDraw

def visualize_detection(images, coordinates, classes_list):
    assert images
    assert coordinates
    assert classes_list
    assert isinstance(classes_list, list)
    assert isinstance(images, np.ndarray)
    assert isinstance(coordinates, np.ndarray)

    images_batch_size, images_channels, images_width, images_height = \
        images.shape

    coordinates_batch_size, coordinates_width, coordinates_height, coordinates_channels = \
        coordinates.shape

    assert images_batch_size == coordinates_batch_size

    for idx in range(images_batch_size):
        image = images[idx, :, :, :]
        coordinates = coordinates[idx, :, :, :]

        image = transforms.ToPILImage()(image)
        image_width, image_height = image.size

        draw = ImageDraw.Draw(image)

        dx = image_width // 7
        dy = image_height // 7

        y_start_point = 0
        y_end_point = image_height

        for i in range(0, image_width, dx):
            y_axis_line = ((i, y_start_point), (i, y_end_point))
            draw.line(y_axis_line, fill="red")

        x_start_point = 0
        x_end_point = image_width

        for i in range(0, image_height, dy):
            x_axis_line = ((x_start_point, i), (x_end_point, i))
            draw.line(x_axis_line, fill="red")

        obj_coord = coordinates[:, :, 0]
        x_shift = coordinates[:, :, 1]
        y_shift = coordinates[:, :, 2]
        w_ratio = coordinates[:, :, 3]
        h_ratio = coordinates[:, :, 4]
        cls = coordinates[:, :, 5]

        for i in range(7):
            for j in range(7):
                if obj_coord[i][j] == 1:
                    x_center = dx * i + int(dx * x_shift[i][j])
                    y_center = dy * j + int(dy * y_shift[i][j])
                    width = int(w_ratio[i][j] * image_width)
                    height = int(h_ratio[i][j] * image_height)

                    xmin = x_center - (width // 2)
                    ymin = y_center - (height // 2)
                    xmax = xmin + width
                    ymax = ymin + height

                    draw.rectangle(((xmin, ymin), (xmax, ymax)), outline="blue")

                    draw.rectangle(((dx * i, dy * j), (dx * i + dx, dy * j + dy)), outline='#00ff88')
                    draw.ellipse(((x_center - 2, y_center - 2),
                                  (x_center + 2, y_center + 2)),
                                 fill='blue')
                    draw.text((dx * i, dy * j), classes_list[int(cls[i][j])])




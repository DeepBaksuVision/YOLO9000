from data_loader.voc_dataset import VocDetection
from utils.label_visualizer import visualize_detection_label
from utils.augmentator import Augmenter
import torchvision.transforms as transforms
from imgaug import augmenters as iaa


def main():
    root = "/home/martin/Documents/dev/Deepbaksu_vision/Datasets/VOCdevkit/VOC2007"

    # Dataset class demo
    voc = VocDetection(root)
    for i in range(10):
        image, target = voc.__getitem__(i)
        print(image, target)
        visualize_detection_label(image, target, voc.classes_list, (13, 13))

    # Augmentation Demo
    seq = iaa.SomeOf(2, [
                iaa.Multiply((1.2, 1.5)),  # change brightness, doesn't affect BBs
                iaa.Affine(
                    translate_px={"x": 3, "y": 10},
                    scale=(0.9, 0.9)
                ),  # translate by 40/60px on x/y axis, and scale to 50-70%, affects BBs
                iaa.AdditiveGaussianNoise(scale=0.1 * 255),
                iaa.CoarseDropout(0.02, size_percent=0.15, per_channel=0.5),
                iaa.Affine(rotate=(-45, 40)),
                iaa.Sharpen(alpha=0.5)
    ])

    composed = transforms.Compose([Augmenter(seq)])

    voc = VocDetection(root, transform=composed)
    for i in range(10):
        image, target = voc.__getitem__(i)
        print(image, target)
        visualize_detection_label(image, target, voc.classes_list, (13, 13))
if __name__ == "__main__":
    main()

from model.model import Yolo9000

yolo9000 = Yolo9000(num_classes=20,
                    num_prior_boxes=5,
                    prior_boxes=[[1., 1.], [2., 2.], [3., 3.], [4., 4.], [5., 5.]],
                    device="cpu",
                    input_size=(416, 416))

model = yolo9000
model.summary()

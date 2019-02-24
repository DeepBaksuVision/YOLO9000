import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


def nll_loss(output, target):
    return F.nll_loss(output, target)


class Loss():
    def __init__(self):
        self.layer1 = None
        self.layer2 = None
        self.layer3 = None
        self.layer4 = None
        self.layer5 = None
        self.layer6 = None
        self.layer7 = None

        self.global_average_pooling = torch.nn.AvgPool2d(1)

        self.bbox_loss = None
        self.iou_loss = None
        self.cls_loss_loss = None

        self.label_name = ('')
        self.anchors = np.array([], dtype=np.float)

        self.num_classses = len(self.label_name)
        self.num_anchors = len(self.anchors)

    def forward(self, data, im_data, gt_boxes=None, gt_classes=None, dontcare=None, size_index=0):

        x = data[0]
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)

        global_average_pool = self.global_average_pooling(out)

        batch_size, _, h, w = global_average_pool.size()
        global_average_pool_reshaped = \
            global_average_pool.permute(0, 2, 3, 1).contiguous().view(batch_size,
                                                                      -1, self.num_anchors, self.num_classes + 5)

        xy_pred = F.sigmoid(global_average_pool_reshaped[:, :, :, 0:2])
        wh_pred = torch.exp(global_average_pool_reshaped[:, :, :, 2:4])
        bbox_pred = torch.cat([xy_pred, wh_pred], 3)
        iou_pred = F.sigmoid(global_average_pool_reshaped[:, :, :, 4:5])

        score_pred = global_average_pool_reshaped[:, :, :, 5:].contiguous()
        prob_pred = F.softmax(score_pred.view(-1, score_pred.size()[-1])).view_as(score_pred)

        bbox_pred_np = bbox_pred.data.cpu().numpy()
        iou_pred_np = iou_pred.data.cpu().numpy()

        _boxes, _ious, _classes, _box_mask, _iou_mask, _class_mask = _build_target(bbox_pred_np,
                                                                                   gt_boxes,
                                                                                   gt_classes,
                                                                                   dontcare,
                                                                                   iou_pred_np,
                                                                                   size_index)

        _boxes = Variable(torch.from_numpy(Variable(torch.from_numpy(_boxes).type(torch.FloatTensor), volatile=False)))
        _ious = Variable(torch.from_numpy(Variable(torch.from_numpy(_ious).type(torch.FloatTensor), volatile=False)))
        _classes = Variable(torch.from_numpy(Variable(torch.from_numpy(_classes).type(torch.FloatTensor), volatile=False)))

        num_boxes = sum((len(boxes) for boxes in gt_boxes))

        # _boxes[:, :, :, 2:4] = torch.log(_boxes[:, :, :, 2:4])
        box_mask = box_mask.expand_as(_boxes)
        class_mask = class_mask.expand_as(prob_pred)

        self.bbox_loss = nn.MSELoss(size_average=False)(bbox_pred * box_mask, _boxes * box_mask) / num_boxes
        self.iou_loss = nn.MSELoss(size_average=False)(iou_pred * iou_mask, _ious * iou_mask) / num_boxes
        self.cls_loss = nn.MSELoss(size_average=False)(prob_pred * class_mask, _classes * class_mask) / num_boxes

        return bbox_pred, iou_pred, prob_pred


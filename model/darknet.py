import torch
import json
from torch import nn
import torch.nn.functional as F
from base import BaseModel

model_paths = {
    'darknet19': 'model_zoo/darknet19-visionNoob.pth'
}

class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        N = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        x = F.avg_pool2d(x, (H, W))
        x = x.view(N, C)
        return x

def make_layers(json_path):
    model_config = json.load(open("darknet19.json"))
    models = torch.nn.ModuleList()
    conv_id = 0
    channel_in = 3
    for block in model_config:        
        if not 'type' in model_config[block]:
            continue
        if model_config[block]['type'] == 'convolutional_3x3':
            conv_id = conv_id + 1
            model = nn.Sequential()

            channel_out = model_config[block]['layers']['conv']

            model.add_module('conv{0}'.format(conv_id), nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, bias=False))
            model.add_module('bn{0}'.format(conv_id), nn.BatchNorm2d(channel_out))
            model.add_module('leaky{0}'.format(conv_id), nn.LeakyReLU(0.1, inplace=True))
            models.append(model)
            channel_in = channel_out

        if model_config[block]['type'] == 'convolutional_1x1':
            conv_id = conv_id + 1
            model = nn.Sequential()
            
            channel_out = model_config[block]['layers']['conv']

            model.add_module('conv{0}'.format(conv_id), nn.Conv2d(channel_in, channel_out, kernel_size=1, padding=1, bias=False))
            model.add_module('bn{0}'.format(conv_id), nn.BatchNorm2d(channel_out))
            model.add_module('leaky{0}'.format(conv_id), nn.LeakyReLU(0.1, inplace=True))
            models.append(model)
            channel_in = channel_out

        if model_config[block]['type'] == 'maxpool':
            model = nn.MaxPool2d(kernel_size=2, stride=2)
            models.append(model)

        if model_config[block]['type'] == 'last_convolutional':
            model = nn.Sequential()
            conv_id = conv_id + 1
            model.add_module('conv{0}'.format(conv_id),  nn.Conv2d(1024, 1000, kernel_size=1, stride=1, bias=True))
            models.append(model)
        
        if model_config[block]['type'] == 'avgpool':
            model = GlobalAvgPool2d()
            models.append(model)
        
        if model_config[block]['type'] == 'softmax':
            model = nn.Softmax()
            models.append(model)

    return models

class Darknet(BaseModel):
    def __init__(self, json_path="darknet19.json", pretrained=False):
        super(Darknet, self).__init__()
        self.models = make_layers(json_path)

        if pretrained:            
            self.load_state_dict(torch.load(model_paths['darknet19']))
            
    def forward(self, x):
        for module in self.models:
            x = module(x)
        return x

# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle.nn as nn
from paddle.vision.models import resnet18


class ProjectionNet(nn.Layer):
    def __init__(self, pretrained=True, head_layers=None, num_classes=2):
        super(ProjectionNet, self).__init__()
        if head_layers is None:
            head_layers = [512, 512, 512, 512, 512, 512, 512, 512, 128]
        self.resnet18 = resnet18(pretrained=pretrained)

        last_layer = 512
        sequential_layers = []

        for num_neurons in head_layers:
            sequential_layers.append(nn.Linear(last_layer, num_neurons))
            sequential_layers.append(nn.BatchNorm1D(num_neurons))
            sequential_layers.append(nn.ReLU())
            last_layer = num_neurons

        # the last layer without activation

        head = nn.Sequential(
            *sequential_layers
        )
        self.resnet18.fc = nn.Identity()
        self.head = head
        self.out = nn.Linear(last_layer, num_classes)

    def forward(self, x):
        embeds = self.resnet18(x)
        tmp = self.head(embeds)
        logits = self.out(tmp)
        return embeds, logits

    def freeze_resnet(self):
        # freez full resnet18
        for param in self.resnet18.parameters():
            param.trainable = False

        # unfreeze head:
        for param in self.resnet18.fc.parameters():
            param.trainable = True

    def unfreeze(self):
        # unfreeze all:
        for param in self.parameters():
            param.trainable = True

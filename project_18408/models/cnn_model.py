import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append(r"C:\Users\skarn\OneDrive\Documents\MIT\year_3\18.408 Theoretical Foundations for Deep Learning\generalization-error-final-project")

from project_18408.models.resnet_utils import ResNetBuilder, BasicBlock

class CNNToyModel(ResNetBuilder):
    MAX_POOL = 1
    CONV_STRIDE = 2
    LINEAR = 3
    BLOCK = 4

    def __init__(
        self, 
        depth, 
        input_img_dim, 
        input_num_channels, 
        output_dim, 
        kernel_size=3
    ):
        super().__init__()
        assert depth > 1

        temp_size = input_img_dim
        num_pool = 0
        num_channels = [input_num_channels]
        img_shape_per_pool = [input_img_dim]
        while temp_size > 5:
            temp_size = (temp_size - 1) // 2 + 1
            num_pool += 1
            if num_pool == 1:
                num_channels.append(16)
            else:
                num_channels.append(num_channels[-1]*2)
            img_shape_per_pool.append(temp_size)
        assert num_pool > 0
        # max pool, conv stride 2, max pool, conv stride 2, linear
        """
        Min 0 blocks per pool
        Per pool conv stride 2 or max pool
        Increase num_blocks evenly distributed we reach depth
        Add extra linear layer as needed since number of layers per block is 2

        """
        # Between one pool and another, depth = 1 + 2 * num_blocks
        layer_config = [self.MAX_POOL] * num_pool
        current_depth = 1
        index = num_pool - 1
        while current_depth < depth and index >= 0:
            layer_config[index] = self.CONV_STRIDE
            index -= 1
            current_depth += 1
        block_dist = np.linspace(0, (depth-current_depth)//2, num_pool+1).astype(int)
        print("block_dist=", block_dist)
        block_config = np.diff(block_dist)
        print("block_config=", block_config)
        print("layer_config=", layer_config)
        for blocks in block_config:
            current_depth += blocks * 2
        num_linear = 1
        num_linear += depth - current_depth
        layers_conv = []
        last_channel_idx = 0
        for n in range(num_pool):
            if layer_config[n] == self.MAX_POOL:
                print("== maxpool")
                layers_conv.append(nn.MaxPool2d(kernel_size=kernel_size, stride=2, padding=1))
            else:
                if block_config[n] == 0:
                    layers_conv.append(nn.Conv2d(num_channels[last_channel_idx], num_channels[n+1],
                                        kernel_size=kernel_size, stride=2, padding=1, bias=False))
                    layers_conv.append(nn.BatchNorm2d(num_channels[n+1]))
                    layers_conv.append(nn.ReLU(inplace=True))
                else:
                    res_blocks = self._make_layer(num_channels[last_channel_idx], num_channels[n+1], block_config[n], 2)
                    layers_conv.append(res_blocks)
                last_channel_idx = n+1
            
        assert num_linear <= 2
        pre_linear_dim = num_channels[-1] * img_shape_per_pool[-1]**2
        print(pre_linear_dim)
        layers_fc = []
        if num_linear == 2:   
            layers_fc.append(nn.Linear(pre_linear_dim, pre_linear_dim // 2, bias=True))
            layers_fc.append(nn.ReLU(inplace=True))
            layers_fc.append(nn.Linear(pre_linear_dim // 2, output_dim, bias=True))
        else:
            layers_fc.append(nn.Linear(pre_linear_dim, output_dim, bias=True))
        # print(pre_linear_dim)
        self.layers_conv = nn.Sequential(*layers_conv)
        self.layers_fc = nn.Sequential(*layers_fc)

        # print(layer_config)
        # self._make_layer(self, in_planes, out_planes, num_blocks, stride=1, block=BasicBlock)
        pass
    
    def forward(self, x):
        x = self.layers_conv(x)
        x = x.view(x.size(0), -1)
        x = self.layers_fc(x)
        return x
    
    def get_layers_with_weights(self):
        layers = []
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                layers.append(m)
            elif isinstance(m, nn.Linear):
                layers.append(m)
        return layers


if __name__ == '__main__':
    toy_model = CNNToyModel(40, 28, 1, 10)
    from torchsummary import summary
    print(toy_model)
    summary(toy_model, (1, 28, 28))
    
    print((toy_model.get_layers_with_weights()))
    print(len(toy_model.get_layers_with_weights()))
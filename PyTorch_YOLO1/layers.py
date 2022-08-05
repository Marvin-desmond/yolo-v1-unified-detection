from typing import Callable, Union, Any, Tuple as tuple, List as list
import torch

Conv2d = torch.nn.Conv2d
BatchNorm2d = torch.nn.BatchNorm2d
MaxPool2d = torch.nn.MaxPool2d
LeakyReLU = torch.nn.LeakyReLU
ReLU = torch.nn.ReLU
Dropout = torch.nn.Dropout
Concat = torch.cat

# 1 * 1 Convolution
def One_by_one_Conv(in_channels: int, out_channels: int = 1) -> Callable:
    def apply(x):
        x = Conv2d(in_channels= in_channels, out_channels=out_channels,  kernel_size = 1, 
            stride = 1, padding = 0)(x)
        return x 
    return apply    

# Convolution ~ Batch ~ Leaky ReLU
def Conv_Batch_LeakyReLU(
    in_channels: int, 
    out_channels: int, 
    kernel_size: Union[int, tuple[int, int]],
    stride: Union[int, tuple[int, int]],
    padding: int,
) -> Callable:
    def apply(x):
        x = Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size = kernel_size, 
            stride = stride, padding = padding,
            )(x)
        x = BatchNorm2d(out_channels)(x)
        x = LeakyReLU()(x)
        return x
    return apply

def Dense_Connection(
    dense_architecture: list[Any],
) -> Callable:
    def apply(x):
        num_repeats: int = dense_architecture[-1]
        conv_configs: list[
            tuple[
                int, 
                Union[int, tuple[int, int]], 
                Union[int, tuple[int, int]], 
                str]
            ] = dense_architecture[:-1] 
        for _ in range(num_repeats):
            for conv_config in conv_configs:
                x = Conv_Batch_LeakyReLU(
                    in_channels = conv_config[1], 
                    out_channels = conv_config[2],
                    kernel_size = conv_config[0],
                    stride = conv_config[3], 
                    padding = conv_config[4]
                    )(x)
        return x 
    return apply



# MaxPooling

# DropOut

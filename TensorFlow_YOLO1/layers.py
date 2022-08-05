from typing import Callable, Union, Any, Tuple as tuple, List as list
import tensorflow as tf

Conv2D = tf.keras.layers.Conv2D
BatchNormalization = tf.keras.layers.BatchNormalization
MaxPooling2D = tf.keras.layers.MaxPooling2D
LeakyReLU = tf.keras.layers.LeakyReLU
ReLU = tf.keras.layers.ReLU
Dropout = tf.keras.layers.Dropout
Concatenate = tf.keras.layers.Concatenate

# 1 * 1 Convolution
def One_by_one_Conv(filters: int = 1) -> Callable:
    def apply(x):
        x = Conv2D(
            filters = filters, kernel_size = 1, 
            strides = 1, padding = "same")(x)
        return x 
    return apply    


# Convolution ~ Batch ~ Leaky ReLU
def Conv_Batch_LeakyReLU(
    filters: int,
    kernel_size: Union[int, tuple[int, int]],
    strides: Union[int, tuple[int, int]],
    padding: str,
) -> Callable:
    def apply(x):
        x = Conv2D(
            filters = filters, kernel_size = kernel_size, 
            strides = strides, padding = padding,
            )(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        return x
    return apply

# Dense Connections
def Dense_Block(
    filters: int,
    kernel_size: Union[int, tuple[int, int]] = (3, 3),
    strides: Union[int, tuple[int, int]] = (1, 1),        
    padding: str = "same"
    ) -> Callable:
    def apply(x):
        y = x
        y = BatchNormalization()(y)
        y = ReLU()(y)
        y = Conv2D(
            filters = filters, 
            kernel_size = kernel_size,
            strides = strides,
            padding = padding
        )(y)
        #############################
        # y = Concatenate()([x, y]) #
        #############################
        return y 
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
                ########################
                #  x = Dense_Block(    #
                ########################
                x = Conv_Batch_LeakyReLU(
                    filters = conv_config[1],
                    kernel_size = conv_config[0],
                    strides = conv_config[2], 
                    padding = conv_config[3]
                    )(x)
        return x 
    return apply



# MaxPooling

# DropOut

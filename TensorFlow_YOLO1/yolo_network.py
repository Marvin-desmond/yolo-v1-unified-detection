import layers
import tensorflow as tf
import pprint

Input = tf.keras.Input 

MaxPooling2D = tf.keras.layers.MaxPooling2D
Dropout = tf.keras.layers.Dropout
GlobalAveragePooling2D = tf.keras.layers.GlobalAveragePooling2D
Dense = tf.keras.layers.Dense
Flatten = tf.keras.layers.Flatten 
Reshape = tf.keras.layers.Reshape


def Backbone_Module(input_: tf.keras.layers.Layer) -> tf.keras.layers.Layer:
    """                                                                                                 ################################
    This function builds the backbone architecture for YOLOv1 Network                                   #         YOLO BACKBONE        #
    and returns it to the caller as a Functional API model,                                             #         ARCHITECTURE         #
    i.e. tf.keras Model                                                                                 ################################ 
    """                                                                                                 #                              #
    x = layers.Conv_Batch_LeakyReLU(filters=64, kernel_size=7, strides=2,  padding="same")(input_)      #     (7, 64, 2, "same")       # 
    x = MaxPooling2D(pool_size = 2, strides = 2)(x)                                                     #             "M"              #
    x = layers.Conv_Batch_LeakyReLU(filters = 192, kernel_size = 3, strides = 1, padding = "same")(x)   #     (3, 192, 1, "same")      #
    x = MaxPooling2D(pool_size = 2, strides = 2)(x)                                                     #             "M"              #
    x = layers.One_by_one_Conv(filters = 128)(x)                                                        #             128              #  
    x = layers.Conv_Batch_LeakyReLU(filters = 256, kernel_size = 3, strides = 1, padding = "same")(x)   #     (3, 256, 1, "same")      #
    x = layers.One_by_one_Conv(filters = 256)(x)                                                        #             256              #
    x = layers.Conv_Batch_LeakyReLU(filters = 512, kernel_size = 3, strides = 1, padding = "same")(x)   #     (3, 512, 1, "same")      #
    x = MaxPooling2D(pool_size = 2, strides = 2)(x)                                                     #             "M"              #
    x = layers.Dense_Connection([(1, 256, 1, "valid"), (3, 512, 1, "same"), 4])(x)                      #  1,256,1,"V"~3,512,1,"S"*4   #
    x = layers.One_by_one_Conv(filters = 512)(x)                                                        #             512              #
    x = layers.Conv_Batch_LeakyReLU(filters = 1024, kernel_size = 3, strides = 1, padding = "same")(x)  #     (3, 1024, 1, "same")     #
    x = MaxPooling2D(pool_size = 2, strides = 2)(x)                                                     #             "M"              #
    x = layers.Dense_Connection([(1, 512, 1, "valid"), (3, 1024, 1, "same"), 2])(x)                     #  1,512,1,"V"~3,1024,1,"S"*2  #
    x = layers.Conv_Batch_LeakyReLU(filters = 1024, kernel_size = 3, strides = 1, padding = "same")(x)  #     (3, 1024, 1, "same")     #
    x = layers.Conv_Batch_LeakyReLU(filters = 1024, kernel_size = 3, strides = 2, padding = "same")(x)  #     (3, 1024, 2, "same")     #
    x = layers.Conv_Batch_LeakyReLU(filters = 1024, kernel_size = 3, strides = 1, padding = "same")(x)  #     (3, 1024, 1, "same")     #
    x = layers.Conv_Batch_LeakyReLU(filters = 1024, kernel_size = 3, strides = 1, padding = "same")(x)  #     (3, 1024, 1, "same")     #
                                                                                                        #                              #
                                                                                                        ################################
    return x



def Head_Module(input_: tf.keras.layers.Layer, S: int = 7, B: int = 2, C: int = 20) -> tf.keras.layers.Layer:
    x = GlobalAveragePooling2D()(input_)
    x = Flatten()(x)
    x = Dense(S * S * (B * 5 + C), activation="relu")(x)
    x = Dropout(rate = 0.3)(x)
    output_ = Reshape((S, S, (B * 5 + C)))(x)
    return output_


def YOLO_Network(input_shape: tuple[int, int, int] = (448, 448, 3), use_pretrained: bool = False) -> tf.keras.Model:
    ####### RESNET MODULE
    def Resnet_Module(input_: tf.keras.layers.Layer) -> tf.keras.layers.Layer:
        pretrained_resnet50 = tf.keras.applications.resnet_v2.ResNet50V2(include_top=False, weights='imagenet')
        pretrained_resnet50.trainable = False  # Freeze the outer model
        assert pretrained_resnet50.trainable == False
        # Now for Pretrained backBone module
        x = pretrained_resnet50(input_)
        # To align with ======== None, 7, 7, 1024 ========
        # Let's do 1 x 1 convolutions and max pooling 2d
        x = layers.One_by_one_Conv(filters = 1024)(x)
        x = MaxPooling2D(pool_size = 2, strides = 2)(x)
        return x
    # WHERE ALL THE ACTION TAKES PLACE
    input_ = Input(shape = input_shape)
    if use_pretrained:
        # Single liner composition 
        Preprocessor = (lambda: (lambda x: tf.keras.applications.resnet_v2.preprocess_input(x) ))
        x = Preprocessor()(input_)
        x = Resnet_Module(x)
    else:
        mean = [0.229, 0.224, 0.225]
        std = [0.229, 0.224, 0.225]
        x  = tf.keras.layers.Normalization(
            mean = mean,
            variance = [round(i**2, 2) for i in std]
        )(input_ / 255.0)
        x = Backbone_Module(x)
    x = Head_Module(x)
    return tf.keras.Model(inputs = [input_], outputs = [x])

if __name__ == "__main__":
    yolo = YOLO_Network(use_pretrained = False)
    in_ = tf.random.uniform((1, 448, 448, 3))
    out_ = yolo(in_)
    print(out_.shape)
    # print(yolo.summary())
    for layer in yolo.layers:
        if "conv2d" in layer.name:
            print(
                (" ".join(layer.name.split("_")[:-1])) +
                    f"\tOUTPUT: {layer.output_shape}" + 
                    f"\tPARAMS: {layer.count_params()}") 




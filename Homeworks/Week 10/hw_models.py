import tensorflow as tf

import tensorflow.keras.models as km
import tensorflow.keras.layers as kl
import tensorflow.keras.regularizers as kr

def cnn_model(imsize, stride, kernelsize,
              n_channels=1, num_of_filters=16, reg=5e-5,
              padding='same', activation='relu', n_class=10,
              model_name='default_model'):
    """
    Using the modified version of Dezső Ribli's CNN, which I've used
    for my BSc thesis work. The network waits for square-shaped images
    as an input.
    
    Parameters
    ----------
    imsize : int
        Side length of the square-shaped input image in pixels.
    stride : int
        Stepsize of the convolutional kernel.
    kernelsize : int
        Size of the convolutional kernel.
    n_channels : int
        Number of color/other channels of the input images.

    Returns
    -------
    model : tensorflow.python.keras.engine.training.Model
        The constructed, yet unconfigured CNN model.
    """
    # Tensorflow placeholder for inputs
    inp = kl.Input(shape=(imsize, imsize, n_channels))

    #
    # Convolutional block 1.
    # 3x3CONVx16 -> ReLU -> 3x3CONVx16 -> ReLU -> MAXPOOLx2
    #
    x = kl.Conv2D(filters=num_of_filters,                   # 3x3CONVx16
                  kernel_size=(kernelsize, kernelsize),
                  padding=padding,
                  strides=(stride, stride),
                  kernel_regularizer=kr.l2(reg))(inp)
    x = kl.Activation(activation)(kl.BatchNormalization()(x))   # ReLU

    x = kl.Conv2D(filters=num_of_filters,                   # 3x3CONVx16
                  kernel_size=(kernelsize, kernelsize),
                  padding=padding,
                  strides=(stride, stride),
                  kernel_regularizer=kr.l2(reg))(x)
    x = kl.Activation(activation)(kl.BatchNormalization()(x))   # ReLU

    x = kl.MaxPooling2D(strides=(2, 2))(x)                  # MAXPOOLx2


    #
    # Convolutional block 2.
    # 3x3CONVx32 -> ReLU -> 3x3CONVx32 -> ReLU -> MAXPOOLx2
    #
    x = kl.Conv2D(filters=2*num_of_filters,                 # 3x3CONVx32
                  kernel_size=(kernelsize, kernelsize),
                  padding=padding,
                  strides=(stride, stride),
                  kernel_regularizer=kr.l2(reg))(x)
    x = kl.Activation(activation)(kl.BatchNormalization()(x))   # ReLU

    x = kl.Conv2D(filters=2*num_of_filters,                 # 3x3CONVx32
                  kernel_size=(kernelsize, kernelsize),
                  padding=padding,
                  strides=(stride, stride),
                  kernel_regularizer=kr.l2(reg))(x)
    x = kl.Activation(activation)(kl.BatchNormalization()(x))   # ReLU
    
    x = kl.MaxPooling2D(strides=(2, 2))(x)                  # MAXPOOLx2

    # End of convolution
    x = kl.GlobalAveragePooling2D()(x)                      # AVGPOOL

    # Final flatten FC
    x = kl.Dense(units=n_class,
                 activation='softmax',
                 name='final_dense')(x)

    # Define model
    model = km.Model(inputs=inp, outputs=x,
                     name=model_name)

    # Multi GPU model
    #if(len(gpu.split(',')) > 1):
    #    model = multi_gpu_model(model, gpus=len(gpu.split(',')))

    return model

def better_cnn_model(imsize, stride, kernelsize,
                     n_channels=1, num_of_filters=32, reg=5e-5,
                     padding='same', activation='relu', n_class=10,
                     model_name='default_model'):
    """
    Using the modified version of Dezső Ribli's CNN, which I've used
    for my BSc thesis work. The network waits for square-shaped images
    as an input.
    
    Parameters
    ----------
    imsize : int
        Side length of the square-shaped input image in pixels.
    stride : int
        Stepsize of the convolutional kernel.
    kernelsize : int
        Size of the convolutional kernel.
    n_channels : int
        Number of color/other channels of the input images.

    Returns
    -------
    model : tensorflow.python.keras.engine.training.Model
        The constructed, yet unconfigured CNN model.
    """
    # Tensorflow placeholder for inputs
    inp = kl.Input(shape=(imsize, imsize, n_channels))

    #
    # Convolutional block 1.
    # 3x3CONVx32 -> ReLU -> 3x3CONVx32 -> ReLU -> MAXPOOLx2
    #
    x = kl.Conv2D(filters=num_of_filters,                   # 3x3CONVx32
                  kernel_size=(kernelsize, kernelsize),
                  padding=padding,
                  strides=(stride, stride),
                  kernel_regularizer=kr.l2(reg))(inp)
    x = kl.Activation(activation)(kl.BatchNormalization()(x))   # ReLU

    x = kl.Conv2D(filters=num_of_filters,                   # 3x3CONVx32
                  kernel_size=(kernelsize, kernelsize),
                  padding=padding,
                  strides=(stride, stride),
                  kernel_regularizer=kr.l2(reg))(x)
    x = kl.Activation(activation)(kl.BatchNormalization()(x))   # ReLU

    x = kl.MaxPooling2D(strides=(2, 2))(x)                  # MAXPOOLx2


    #
    # Convolutional block 2.
    # 3x3CONVx64 -> ReLU -> 3x3CONVx64 -> ReLU -> MAXPOOLx2
    #
    x = kl.Conv2D(filters=2*num_of_filters,                 # 3x3CONVx64
                  kernel_size=(kernelsize, kernelsize),
                  padding=padding,
                  strides=(stride, stride),
                  kernel_regularizer=kr.l2(reg))(x)
    x = kl.Activation(activation)(kl.BatchNormalization()(x))   # ReLU

    x = kl.Conv2D(filters=2*num_of_filters,                 # 3x3CONVx64
                  kernel_size=(kernelsize, kernelsize),
                  padding=padding,
                  strides=(stride, stride),
                  kernel_regularizer=kr.l2(reg))(x)
    x = kl.Activation(activation)(kl.BatchNormalization()(x))   # ReLU
    
    x = kl.MaxPooling2D(strides=(2, 2))(x)                  # MAXPOOLx2
    
    
    #
    # Convolutional block 3.
    # 3x3CONVx128 -> ReLU -> 3x3CONVx128 -> ReLU -> MAXPOOLx2
    #
    x = kl.Conv2D(filters=4*num_of_filters,                 # 3x3CONVx128
                  kernel_size=(kernelsize, kernelsize),
                  padding=padding,
                  strides=(stride, stride),
                  kernel_regularizer=kr.l2(reg))(x)
    x = kl.Activation(activation)(kl.BatchNormalization()(x))   # ReLU

    x = kl.Conv2D(filters=4*num_of_filters,                 # 3x3CONVx128
                  kernel_size=(kernelsize, kernelsize),
                  padding=padding,
                  strides=(stride, stride),
                  kernel_regularizer=kr.l2(reg))(x)
    x = kl.Activation(activation)(kl.BatchNormalization()(x))   # ReLU
    
    x = kl.MaxPooling2D(strides=(2, 2))(x)                  # MAXPOOLx2

    # End of convolution
    x = kl.GlobalAveragePooling2D()(x)                      # AVGPOOL

    # Final flatten FC
    x = kl.Dense(units=n_class,
                 activation='softmax',
                 name='final_dense')(x)

    # Define model
    model = km.Model(inputs=inp, outputs=x,
                     name=model_name)

    # Multi GPU model
    #if(len(gpu.split(',')) > 1):
    #    model = multi_gpu_model(model, gpus=len(gpu.split(',')))

    return model
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model

def convolution_block(input_layer, num_filters):
    """Defines a convolution block consisting of two convolution layers followed by batch normalization and ReLU activation.

    Args:
        input_layer (tensor): Input tensor.
        num_filters (int): Number of filters for convolution layers.

    Returns:
        tensor: Output tensor.
    """
    x = Conv2D(num_filters, 3, padding="same")(input_layer)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def encoder_block(input_layer, num_filters):
    """Defines an encoder block consisting of a convolution block followed by max pooling.

    Args:
        input_layer (tensor): Input tensor.
        num_filters (int): Number of filters for convolution layers.

    Returns:
        tuple: Output tensor of convolution block and max pooled tensor.
    """
    x = convolution_block(input_layer, num_filters)
    pooled_layer = MaxPool2D((2, 2))(x)
    return x, pooled_layer

def decoder_block(input_layer, skip_features, num_filters):
    """Defines a decoder block consisting of transpose convolution, concatenation with skip connection, and convolution block.

    Args:
        input_layer (tensor): Input tensor.
        skip_features (tensor): Skip connection tensor.
        num_filters (int): Number of filters for convolution layers.

    Returns:
        tensor: Output tensor.
    """
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input_layer)
    x = Concatenate()([x, skip_features])
    x = convolution_block(x, num_filters)
    return x

def build_unet(input_shape):
    """Builds the U-Net model architecture.

    Args:
        input_shape (tuple): Shape of input tensor.

    Returns:
        Model: U-Net model.
    """
    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    bottleneck = convolution_block(p4, 1024)

    d1 = decoder_block(bottleneck, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    model = Model(inputs, outputs, name="U-Net")
    return model

if __name__ == "__main__":
    input_shape = (512, 512, 3)
    model = build_unet(input_shape)
    model.summary()


from keras.layers import *
from keras.optimizers import *
from keras.models import *
from layer import *
def RCAUNet(input_size=(512, 512, 3), start_neurons=16):

    inputs = Input(input_size)
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation='relu', padding="same")(inputs)
    conv1 =RCAB(conv1,16)
    pool1 = MaxPooling2D((2, 2))(conv1)



    conv2 = Conv2D(start_neurons * 2, (3, 3), activation='relu', padding="same")(pool1)
    conv2 =RCAB(conv2,16)
    pool2 = MaxPooling2D((2, 2))(conv2)


    conv3 = Conv2D(start_neurons * 4, (3, 3), activation='relu', padding="same")(pool2)
    conv3 =RCAB(conv3,16)
    pool3 = MaxPooling2D((2, 2))(conv3)


    conv4 = Conv2D(start_neurons * 8, (3, 3), activation='relu', padding="same")(pool3)
    conv4 =RCAB(conv4,16)
    pool4 = MaxPooling2D((2, 2))(conv4)


    convm = Conv2D(start_neurons * 16, (3, 3), activation='relu', padding="same")(pool4)
    convm =RCAB(convm,16)


    deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = concatenate([deconv4, conv4])


    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation='relu', padding="same")(uconv4)
    uconv4 =RCAB(uconv4,16)


    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    uconv3 = concatenate([deconv3, conv3])

    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation='relu', padding="same")(uconv3)
    uconv3 =RCAB(uconv3,16)


    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])

    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation='relu', padding="same")(uconv2)
    uconv2 =RCAB(uconv2,16)


    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])


    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation='relu', padding="same")(uconv1)
    uconv1 =RCAB(uconv1,16)

    output_layer_noActi = Conv2D(1, (1, 1), padding="same", activation=None)(uconv1)
    output_layer = Activation('sigmoid')(output_layer_noActi)

    model = Model(input=inputs, output=output_layer)

    model.compile(optimizer=Adam(lr=1e-3), loss='binary_crossentropy', metrics=['accuracy'])

    return model



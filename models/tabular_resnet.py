from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, Activation, Add, Dropout


def create_robust_resnet(input_dim, n_classes, n_layers=4, h_dim=256):

    x_in = Input(shape=(input_dim,))
    h = x_in

    h = Dense(h_dim)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    for i in range(n_layers):
        shortcut = h

        h = Dense(h_dim)(h)
        h = BatchNormalization()(h)
        h = Activation('relu')(h)
        h = Dropout(0.2)(h)

        h = Dense(h_dim)(h)
        h = BatchNormalization()(h)

        h = Add()([h, shortcut])
        h = Activation('relu')(h)

    out = Dense(n_classes, activation='softmax')(h)

    return Model(x_in, out)
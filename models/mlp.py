from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Activation, Dropout


def create_mlp(input_dim, n_classes, h_dim=128, n_layers=2, dropout_rate=0.2):
    model = Sequential()

    model.add(Dense(h_dim, input_dim=input_dim))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout_rate))

    for _ in range(n_layers - 1):
        model.add(Dense(h_dim))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(dropout_rate))

    model.add(Dense(n_classes, activation='softmax'))

    return model
    
def create_wide_mlp(input_dim, n_classes, dropout_rate=0.3):
    model = Sequential()

    model.add(Dense(512, input_dim=input_dim))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout_rate))

    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout_rate))

    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout_rate))

    model.add(Dense(n_classes, activation='softmax'))

    return

def create_residual_mlp_with_embedding(input_dim, n_classes, dropout_rate=0.2, emb_dim=128):
    inputs = Input(shape=(input_dim,), name="input")

    x = Dense(512)(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    shortcut = x
    x = Dense(512)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(512)(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut])
    x = LeakyReLU(alpha=0.1)(x)

    shortcut = x
    x = Dense(512)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(512)(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut])
    x = LeakyReLU(alpha=0.1)(x)

    x = Dense(256)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    embedding = Dense(emb_dim, name="embedding")(x)
    embedding_bn = BatchNormalization()(embedding)
    embedding_act = LeakyReLU(alpha=0.1, name="embedding_act")(embedding_bn)

    outputs = Dense(n_classes, activation='softmax', name="softmax")(embedding_act)

    clf_model = Model(inputs=inputs, outputs=outputs)
    emb_model = Model(inputs=inputs, outputs=embedding_act)
    return clf_model, emb_model

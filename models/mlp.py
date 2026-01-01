from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Activation, Dropout


def create_mlp(input_dim, n_classes, h_dim=128, n_layers=2, dropout_rate=0.2):
    """
    Creates a flexible MLP for tabular data with regularization.
    """
    model = Sequential()

    # Input layer
    model.add(Dense(h_dim, input_dim=input_dim))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout_rate))

    # Hidden layers
    for _ in range(n_layers - 1):
        model.add(Dense(h_dim))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(dropout_rate))

    # Output layer
    model.add(Dense(n_classes, activation='softmax'))

    return model
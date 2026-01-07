from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, Activation, Add, Dropout


def create_robust_resnet(input_dim, n_classes, n_layers=4, h_dim=256):
    """
    Ein Residual Network (ResNet) für Tabellendaten.

    Warum ResNet?
    Normale neuronale Netze (MLPs) haben oft Probleme, sehr chaotische Daten ('Normal')
    zu lernen, ohne dabei die Struktur von 'Fraud' zu vergessen.
    Die Skip-Connections (Add-Layer) erlauben dem Netzwerk, Informationen
    durchzureichen, was das Training stabilisiert und vertieft.
    """
    x_in = Input(shape=(input_dim,))
    h = x_in

    # Erste Projektion / Feature Extraction
    h = Dense(h_dim)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    # Residual Blocks
    for i in range(n_layers):
        shortcut = h  # Merken für die Skip-Connection

        # --- Block Start ---
        h = Dense(h_dim)(h)
        h = BatchNormalization()(h)
        h = Activation('relu')(h)
        h = Dropout(0.2)(h)  # Wichtig: Dropout verhindert Overfitting auf 'Normal'

        h = Dense(h_dim)(h)
        h = BatchNormalization()(h)
        # --- Block Ende ---

        # Der Residual-Trick: Addiere das Originalsignal (shortcut) wieder dazu
        h = Add()([h, shortcut])
        h = Activation('relu')(h)

    # Output Layer (Klassifikation der Transformation)
    out = Dense(n_classes, activation='softmax')(h)

    return Model(x_in, out)
from glob import glob
import os
import numpy as np
import cv2
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.datasets import mnist, fashion_mnist, cifar100, cifar10
from keras.backend import cast_to_floatx


def resize_and_crop_image(input_file, output_side_length, greyscale=False):
    img = cv2.imread(input_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB if not greyscale else cv2.COLOR_BGR2GRAY)
    height, width = img.shape[:2]
    new_height = output_side_length
    new_width = output_side_length
    if height > width:
        new_height = int(output_side_length * height / width)
    else:
        new_width = int(output_side_length * width / height)
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    height_offset = (new_height - output_side_length) // 2
    width_offset = (new_width - output_side_length) // 2
    cropped_img = resized_img[height_offset:height_offset + output_side_length,
    width_offset:width_offset + output_side_length]
    assert cropped_img.shape[:2] == (output_side_length, output_side_length)
    return cropped_img


def normalize_minus1_1(data):
    return 2 * (data / 255.) - 1


def get_channels_axis():
    import keras
    idf = keras.backend.image_data_format()
    if idf == 'channels_first':
        return 1
    assert idf == 'channels_last'
    return 3


def load_fashion_mnist():
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    X_train = normalize_minus1_1(cast_to_floatx(np.pad(X_train, ((0, 0), (2, 2), (2, 2)), 'constant')))
    X_train = np.expand_dims(X_train, axis=get_channels_axis())
    X_test = normalize_minus1_1(cast_to_floatx(np.pad(X_test, ((0, 0), (2, 2), (2, 2)), 'constant')))
    X_test = np.expand_dims(X_test, axis=get_channels_axis())
    return (X_train, y_train), (X_test, y_test)


def load_mnist():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = normalize_minus1_1(cast_to_floatx(np.pad(X_train, ((0, 0), (2, 2), (2, 2)), 'constant')))
    X_train = np.expand_dims(X_train, axis=get_channels_axis())
    X_test = normalize_minus1_1(cast_to_floatx(np.pad(X_test, ((0, 0), (2, 2), (2, 2)), 'constant')))
    X_test = np.expand_dims(X_test, axis=get_channels_axis())
    return (X_train, y_train), (X_test, y_test)


def load_cifar10():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train = normalize_minus1_1(cast_to_floatx(X_train))
    X_test = normalize_minus1_1(cast_to_floatx(X_test))
    return (X_train, y_train), (X_test, y_test)


def load_cifar100(label_mode='coarse'):
    (X_train, y_train), (X_test, y_test) = cifar100.load_data(label_mode=label_mode)
    X_train = normalize_minus1_1(cast_to_floatx(X_train))
    X_test = normalize_minus1_1(cast_to_floatx(X_test))
    return (X_train, y_train), (X_test, y_test)


def load_creditcard():
    filepath = 'data/creditcard.csv'
    print(f"Loading and processing {filepath}...")
    df = pd.read_csv(filepath)

    seconds_in_day = 24 * 60 * 60
    df['Time_Day'] = df['Time'] % seconds_in_day
    df['time_vector_x'] = np.sin(2 * np.pi * df['Time_Day'] / seconds_in_day)
    df['time_vector_y'] = np.cos(2 * np.pi * df['Time_Day'] / seconds_in_day)
    df = df.drop(['Time', 'Time_Day'], axis=1)

    df['Amount'] = np.log1p(df['Amount'])
    df['Amount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1, 1))

    y = df['Class'].values
    x = df.drop(['Class'], axis=1).values

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    print(f"Data ready. Input shape: {x_train.shape}")
    return (x_train, y_train), (x_test, y_test)


def load_ieee_fraud():
    transaction_path = 'data/train_transaction.csv'
    identity_path = 'data/train_identity.csv'

    print(f"Loading IEEE-CIS Dataset...")
    df_trans = pd.read_csv(transaction_path)
    df_id = pd.read_csv(identity_path)
    df = pd.merge(df_trans, df_id, on='TransactionID', how='left')

    target_col = 'isFraud'

    if 'TransactionDT' in df.columns:
        seconds_in_day = 24 * 60 * 60
        df['Time_Day'] = df['TransactionDT'] % seconds_in_day
        df['time_vector_x'] = np.sin(2 * np.pi * df['Time_Day'] / seconds_in_day)
        df['time_vector_y'] = np.cos(2 * np.pi * df['Time_Day'] / seconds_in_day)
        df = df.drop(['TransactionDT', 'Time_Day'], axis=1)

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.factorize(df[col])[0]

    df = df.fillna(-999)

    if 'TransactionAmt' in df.columns:
        df['TransactionAmt'] = np.log1p(df['TransactionAmt'])

    y = df[target_col].values
    cols_to_drop = [target_col]
    if 'TransactionID' in df.columns:
        cols_to_drop.append('TransactionID')
    x = df.drop(cols_to_drop, axis=1).values

    x = StandardScaler().fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    print(f"IEEE-CIS Data ready. Train shape: {x_train.shape}")
    return (x_train, y_train), (x_test, y_test)
    
def load_elliptic():
    print("Loading Elliptic Bitcoin Dataset...")
    df_classes = pd.read_csv('data/elliptic_bitcoin_dataset/elliptic_txs_classes.csv')
    df_features = pd.read_csv('data/elliptic_bitcoin_dataset/elliptic_txs_features.csv', header=None)

    df_features.rename(columns={0: 'txId'}, inplace=True)
    
    df = pd.merge(df_features, df_classes, on='txId', how='left')
    
    df = df[df['class'] != "unknown"].copy()

    df['class'] = df['class'].astype(str)
    df['label'] = df['class'].map({'2': 0, '1': 1})
    
    y = df['label'].values
    drop_cols = ['txId', 'class', 'label']
    x = df.drop(drop_cols, axis=1).values
    
    x = StandardScaler().fit_transform(x)
    
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=42, stratify=y
    )
    
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    
    print(f"Elliptic loaded. Train shape: {x_train.shape}, Frauds: {y_train.sum()}")
    return (x_train, y_train), (x_test, y_test)


def save_roc_pr_curve_data(scores, labels, file_path):
    scores = scores.flatten()
    labels = labels.flatten()

    scores_pos = scores[labels == 1]
    scores_neg = scores[labels != 1]

    truth = np.concatenate((np.zeros_like(scores_neg), np.ones_like(scores_pos)))
    preds = np.concatenate((scores_neg, scores_pos))
    fpr, tpr, roc_thresholds = roc_curve(truth, preds)
    roc_auc = auc(fpr, tpr)

    precision_norm, recall_norm, pr_thresholds_norm = precision_recall_curve(truth, preds)
    pr_auc_norm = auc(recall_norm, precision_norm)

    precision_anom, recall_anom, pr_thresholds_anom = precision_recall_curve(truth, -preds, pos_label=0)
    pr_auc_anom = auc(recall_anom, precision_anom)

    np.savez_compressed(file_path,
                        preds=preds, truth=truth,
                        fpr=fpr, tpr=tpr, roc_thresholds=roc_thresholds, roc_auc=roc_auc,
                        precision_norm=precision_norm, recall_norm=recall_norm,
                        pr_thresholds_norm=pr_thresholds_norm, pr_auc_norm=pr_auc_norm,
                        precision_anom=precision_anom, recall_anom=recall_anom,
                        pr_thresholds_anom=pr_thresholds_anom, pr_auc_anom=pr_auc_anom)


def create_cats_vs_dogs_npz(cats_vs_dogs_path='./'):
    labels = ['cat', 'dog']
    label_to_y_dict = {l: i for i, l in enumerate(labels)}

    def _load_from_dir(dir_name):
        glob_path = os.path.join(cats_vs_dogs_path, dir_name, '*.*.jpg')
        imgs_paths = glob(glob_path)
        images = [resize_and_crop_image(p, 64) for p in imgs_paths]
        x = np.stack(images)
        y = [label_to_y_dict[os.path.split(p)[-1][:3]] for p in imgs_paths]
        y = np.array(y)
        return x, y

    x_train, y_train = _load_from_dir('train')
    x_test, y_test = _load_from_dir('test')

    np.savez_compressed(os.path.join(cats_vs_dogs_path, 'cats_vs_dogs.npz'),
                        x_train=x_train, y_train=y_train,
                        x_test=x_test, y_test=y_test)


def load_cats_vs_dogs(cats_vs_dogs_path='./'):
    npz_file = np.load(os.path.join(cats_vs_dogs_path, 'cats_vs_dogs.npz'))
    x_train = normalize_minus1_1(cast_to_floatx(npz_file['x_train']))
    y_train = npz_file['y_train']
    x_test = normalize_minus1_1(cast_to_floatx(npz_file['x_test']))
    y_test = npz_file['y_test']

    return (x_train, y_train), (x_test, y_test)


def get_class_name_from_index(index, dataset_name):
    # Dictionary mit Klassennamen
    ind_to_name = {
        'cifar10': ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'),
        'cifar100': ('aquatic mammals', 'fish', 'flowers', 'food containers', 'fruit and vegetables',
                     'household electrical devices', 'household furniture', 'insects', 'large carnivores',
                     'large man-made outdoor things', 'large natural outdoor scenes', 'large omnivores and herbivores',
                     'medium-sized mammals', 'non-insect invertebrates', 'people', 'reptiles', 'small mammals', 'trees',
                     'vehicles 1', 'vehicles 2'),
        'fashion-mnist': ('t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag',
                          'ankle-boot'),
        'cats-vs-dogs': ('cat', 'dog'),
        'creditcard': ('Normal', 'Fraud'),
        'ieee-fraud': ('Normal', 'Fraud'),
        'elliptic': ('Licit', 'Illicit')  # <--- NEU: Elliptic Klassen
    }

    if dataset_name.startswith('elliptic'):
        lookup_key = 'elliptic'
    else:
        lookup_key = dataset_name

    return ind_to_name[lookup_key][index]

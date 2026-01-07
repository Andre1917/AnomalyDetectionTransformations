import os
import csv
from collections import defaultdict
from glob import glob
from datetime import datetime
from multiprocessing import Manager, freeze_support, Process
import numpy as np
import scipy.stats
from scipy.special import psi, polygamma
from sklearn.metrics import roc_auc_score
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import ParameterGrid
from sklearn.externals.joblib import Parallel, delayed
from keras.models import Model, Input, Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical

# --- Custom Modules ---
from utils import load_cifar10, load_cats_vs_dogs, load_fashion_mnist, load_cifar100, load_creditcard, load_ieee_fraud
from utils import save_roc_pr_curve_data, get_class_name_from_index, get_channels_axis
from transformations import Transformer, TabularTransformer
from models.wide_residual_network import create_wide_residual_network
from models.mlp import create_mlp
from models.encoders_decoders import conv_encoder, conv_decoder
from models import dsebm, dagmm, adgan
import keras.backend as K

RESULTS_DIR = 'results_test'  # Separate folder for test results


# --- 1. OUR METHOD (Fixes included) ---
def _transformations_experiment(dataset_load_fn, dataset_name, single_class_ind, gpu_q):
    gpu_to_use = gpu_q.get()
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_to_use
    (x_train, y_train), (x_test, y_test) = dataset_load_fn()

    is_tabular = dataset_name in ['creditcard', 'ieee-fraud']
    if is_tabular:
        transformer = TabularTransformer(n_transforms=8)
        mdl = create_mlp(input_dim=x_train.shape[1], n_classes=transformer.n_transforms)
    else:
        transformer = Transformer(8, 8)
        mdl = create_wide_residual_network(x_train.shape[1:], transformer.n_transforms, 10, 4)

    mdl.compile('adam', 'categorical_crossentropy', ['acc'])
    x_train_task = x_train[y_train.flatten() == single_class_ind]
    transformations_inds = np.tile(np.arange(transformer.n_transforms), len(x_train_task))
    x_train_task_transformed = transformer.transform_batch(np.repeat(x_train_task, transformer.n_transforms, axis=0),
                                                           transformations_inds)
    mdl.fit(x=x_train_task_transformed, y=to_categorical(transformations_inds), batch_size=128, epochs=1)

    EPSILON = 1e-15

    def calc_approx_alpha_sum(observations):
        N, f = len(observations), np.mean(observations, axis=0)
        f = np.clip(f, EPSILON, 1 - EPSILON)
        obs_safe = np.clip(observations, EPSILON, 1 - EPSILON)
        return (N * (len(f) - 1) * (-psi(1))) / (
                    N * np.sum(f * np.log(f)) - np.sum(f * np.sum(np.log(obs_safe), axis=0)))

    def inv_psi(y, iters=5):
        cond = y >= -2.22
        x = cond * (np.exp(y) + 0.5) + (1 - cond) * -1 / (y - psi(1))
        for _ in range(iters): x = x - (psi(x) - y) / polygamma(1, x)
        return x

    def fixed_point_dirichlet_mle(alpha_init, log_p_hat, max_iter=100):
        alpha_new = alpha_old = alpha_init
        for _ in range(max_iter):
            alpha_new = inv_psi(psi(np.sum(alpha_old)) + log_p_hat)
            if np.sqrt(np.sum((alpha_old - alpha_new) ** 2)) < 1e-7: break
            alpha_old = alpha_new
        return alpha_new

    def dirichlet_normality_score(alpha, p):
        return np.sum((alpha - 1) * np.log(np.clip(p, EPSILON, 1 - EPSILON)), axis=-1)

    scores, observed_data = np.zeros((len(x_test),)), x_train_task
    for t_ind in range(transformer.n_transforms):
        observed_dirichlet = np.clip(
            mdl.predict(transformer.transform_batch(observed_data, [t_ind] * len(observed_data)), batch_size=1024),
            EPSILON, 1 - EPSILON)
        log_p_hat_train = np.log(observed_dirichlet).mean(axis=0)
        alpha_0 = observed_dirichlet.mean(axis=0) * calc_approx_alpha_sum(observed_dirichlet)
        scores += dirichlet_normality_score(fixed_point_dirichlet_mle(alpha_0, log_p_hat_train),
                                            mdl.predict(transformer.transform_batch(x_test, [t_ind] * len(x_test)),
                                                        batch_size=1024))

    scores /= transformer.n_transforms
    if not np.all(np.isfinite(scores)): scores[~np.isfinite(scores)] = np.min(scores[np.isfinite(scores)]) if np.any(
        np.isfinite(scores)) else 0.0

    res_dir = os.path.join(RESULTS_DIR, dataset_name)
    if not os.path.exists(res_dir): os.makedirs(res_dir)
    save_roc_pr_curve_data(scores, y_test.flatten() == single_class_ind,
                           os.path.join(res_dir, f'test_{dataset_name}.npz'))
    gpu_q.put(gpu_to_use)


# --- MINI DATA LOADER FOR TESTING ---
def load_mini_ieee():
    print("!!! TEST RUN: Loading 1000 rows of IEEE dataset !!!")
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    # Load small chunk
    df_trans = pd.read_csv('data/train_transaction.csv', nrows=1000)
    df_id = pd.read_csv('data/train_identity.csv', nrows=1000)
    df = pd.merge(df_trans, df_id, on='TransactionID', how='left')

    # Simple processing
    y = df['isFraud'].values
    df = df.drop(['isFraud', 'TransactionID'], axis=1)
    for col in df.columns:
        if df[col].dtype == 'object': df[col] = pd.factorize(df[col])[0]
    x = StandardScaler().fit_transform(df.fillna(-999).values)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    return (x_train, y_train.reshape(-1, 1)), (x_test, y_test.reshape(-1, 1))


if __name__ == '__main__':
    freeze_support()
    man = Manager()
    q = man.Queue(1)
    q.put("0")

    # Run only 1 transform experiment on the mini dataset
    _transformations_experiment(load_mini_ieee, 'ieee-fraud', 0, q)
    print("Test run successful! NaN and Merge logic works.")
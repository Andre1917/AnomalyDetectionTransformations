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

RESULTS_DIR = 'results'


# --- 1. OUR METHOD (GeoTrans / TabTrans) ---
def _transformations_experiment(dataset_load_fn, dataset_name, single_class_ind, gpu_q=None):
    # gpu_q wird ignoriert, wir nutzen einfach die verfügbare GPU
    print(f"[{datetime.now()}] --- Starting Transformations Experiment for Class {single_class_ind} ---")

    (x_train, y_train), (x_test, y_test) = dataset_load_fn()

    is_tabular = dataset_name in ['creditcard', 'ieee-fraud']

    if is_tabular:
        # WICHTIG: 8 Transformationen für IEEE
        transformer = TabularTransformer(n_transforms=8)
        mdl = create_mlp(input_dim=x_train.shape[1], n_classes=transformer.n_transforms)
    else:
        if dataset_name in ['cats-vs-dogs']:
            transformer = Transformer(16, 16)
            n, k = (16, 8)
        else:
            transformer = Transformer(8, 8)
            n, k = (10, 4)
        mdl = create_wide_residual_network(x_train.shape[1:], transformer.n_transforms, n, k)

    mdl.compile('adam', 'categorical_crossentropy', ['acc'])

    # Train only on normal class
    x_train_task = x_train[y_train.flatten() == single_class_ind]
    transformations_inds = np.tile(np.arange(transformer.n_transforms), len(x_train_task))

    print(f"[{datetime.now()}] Creating transformed batch...")
    x_train_task_transformed = transformer.transform_batch(np.repeat(x_train_task, transformer.n_transforms, axis=0),
                                                           transformations_inds)
    batch_size = 128

    print(f"[{datetime.now()}] Starting Training (fit)...")
    # FIX: Wir setzen Epochen hart auf 5, damit es heute noch fertig wird!
    mdl.fit(x=x_train_task_transformed, y=to_categorical(transformations_inds),
            batch_size=batch_size, epochs=5)

    # --- Normality Scoring Logic (mit NaN-Schutz) ---
    EPSILON = 1e-15

    def calc_approx_alpha_sum(observations):
        N = len(observations)
        f = np.mean(observations, axis=0)
        f = np.clip(f, EPSILON, 1 - EPSILON)
        obs_safe = np.clip(observations, EPSILON, 1 - EPSILON)
        return (N * (len(f) - 1) * (-psi(1))) / (
                N * np.sum(f * np.log(f)) - np.sum(f * np.sum(np.log(obs_safe), axis=0)))

    def inv_psi(y, iters=5):
        cond = y >= -2.22
        x = cond * (np.exp(y) + 0.5) + (1 - cond) * -1 / (y - psi(1))
        for _ in range(iters):
            x = x - (psi(x) - y) / polygamma(1, x)
        return x

    def fixed_point_dirichlet_mle(alpha_init, log_p_hat, max_iter=1000):
        alpha_new = alpha_old = alpha_init
        for _ in range(max_iter):
            alpha_new = inv_psi(psi(np.sum(alpha_old)) + log_p_hat)
            if np.sqrt(np.sum((alpha_old - alpha_new) ** 2)) < 1e-9:
                break
            alpha_old = alpha_new
        return alpha_new

    def dirichlet_normality_score(alpha, p):
        p_safe = np.clip(p, EPSILON, 1 - EPSILON)
        return np.sum((alpha - 1) * np.log(p_safe), axis=-1)

    print(f"[{datetime.now()}] Calculating Scores...")
    scores = np.zeros((len(x_test),))
    observed_data = x_train_task

    for t_ind in range(transformer.n_transforms):
        print(f" - Transform {t_ind + 1}/{transformer.n_transforms}")
        observed_dirichlet = mdl.predict(transformer.transform_batch(observed_data, [t_ind] * len(observed_data)),
                                         batch_size=1024)
        observed_dirichlet = np.clip(observed_dirichlet, EPSILON, 1 - EPSILON)

        log_p_hat_train = np.log(observed_dirichlet).mean(axis=0)
        alpha_sum_approx = calc_approx_alpha_sum(observed_dirichlet)
        alpha_0 = observed_dirichlet.mean(axis=0) * alpha_sum_approx
        mle_alpha_t = fixed_point_dirichlet_mle(alpha_0, log_p_hat_train)

        x_test_p = mdl.predict(transformer.transform_batch(x_test, [t_ind] * len(x_test)),
                               batch_size=1024)
        x_test_p = np.clip(x_test_p, EPSILON, 1 - EPSILON)

        scores += dirichlet_normality_score(mle_alpha_t, x_test_p)

    scores /= transformer.n_transforms
    labels = y_test.flatten() == single_class_ind

    if not np.all(np.isfinite(scores)):
        print(f"WARNING: NaNs found. Fixing...")
        finite_scores = scores[np.isfinite(scores)]
        min_val = np.min(finite_scores) if len(finite_scores) > 0 else 0.0
        scores[~np.isfinite(scores)] = min_val

    # Save results
    res_dir = os.path.join(RESULTS_DIR, dataset_name)
    if not os.path.exists(res_dir): os.makedirs(res_dir)

    res_file_name = '{}_transformations_{}_{}.npz'.format(dataset_name,
                                                          get_class_name_from_index(single_class_ind, dataset_name),
                                                          datetime.now().strftime('%Y-%m-%d-%H%M'))
    save_roc_pr_curve_data(scores, labels, os.path.join(res_dir, res_file_name))
    mdl.save_weights(os.path.join(res_dir, res_file_name.replace('.npz', '_weights.h5')))
    print(f"[{datetime.now()}] Saved result: {res_file_name}")


# --- 2. BENCHMARK: Isolation Forest ---
def _train_if_and_score(params, xtrain, test_labels, xtest):
    clf = IsolationForest(**params, n_jobs=1, random_state=42).fit(xtrain)
    return roc_auc_score(test_labels, -clf.decision_function(xtest))

def _isolation_forest_experiment(dataset_load_fn, dataset_name, single_class_ind):
    (x_train, y_train), (x_test, y_test) = dataset_load_fn()
    x_train = x_train.reshape((len(x_train), -1))
    x_test = x_test.reshape((len(x_test), -1))
    x_train_task = x_train[y_train.flatten() == single_class_ind]

    # Subsampling for large datasets like IEEE
    if len(x_train_task) > 100000 and dataset_name == 'ieee-fraud':
         subsample_inds = np.random.choice(len(x_train_task), 100000, replace=False)
         x_train_task = x_train_task[subsample_inds]
         print("Subsampling Isolation Forest to 100k.")

    pg = ParameterGrid({'contamination': [0.01, 0.05, 0.1], 'n_estimators': [100]})

    results = Parallel(n_jobs=4)(
        delayed(_train_if_and_score)(d, x_train_task, y_test.flatten() == single_class_ind, x_test)
        for d in pg)

    best_params, _ = max(zip(pg, results), key=lambda t: t[-1])
    best_clf = IsolationForest(**best_params, n_jobs=1, random_state=42).fit(x_train_task)
    scores = -best_clf.decision_function(x_test)
    labels = y_test.flatten() == single_class_ind

    res_dir = os.path.join(RESULTS_DIR, dataset_name)
    if not os.path.exists(res_dir): os.makedirs(res_dir)

    res_file_name = '{}_isolation-forest_{}_{}.npz'.format(dataset_name,
                                                           get_class_name_from_index(single_class_ind, dataset_name),
                                                           datetime.now().strftime('%Y-%m-%d-%H%M'))
    save_roc_pr_curve_data(scores, labels, os.path.join(res_dir, res_file_name))


# --- 3. BENCHMARK: Autoencoder ---
def _tabular_autoencoder_experiment(dataset_load_fn, dataset_name, single_class_ind, gpu_q):
    gpu_to_use = gpu_q.get()
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_to_use

    (x_train, y_train), (x_test, y_test) = dataset_load_fn()
    x_train_task = x_train[y_train.flatten() == single_class_ind]

    input_dim = x_train.shape[1]
    encoding_dim = int(input_dim / 2)

    model = Sequential()
    model.add(Dense(encoding_dim, activation="relu", input_shape=(input_dim,)))
    model.add(Dense(input_dim, activation="linear"))
    model.compile(optimizer='adam', loss='mse')

    model.fit(x_train_task, x_train_task, epochs=50, batch_size=128, verbose=0)

    reconstructions = model.predict(x_test)
    scores = np.mean(np.power(x_test - reconstructions, 2), axis=1)
    labels = y_test.flatten() == single_class_ind

    res_dir = os.path.join(RESULTS_DIR, dataset_name)
    if not os.path.exists(res_dir): os.makedirs(res_dir)

    res_file_name = '{}_autoencoder_{}_{}.npz'.format(dataset_name,
                                                      get_class_name_from_index(single_class_ind, dataset_name),
                                                      datetime.now().strftime('%Y-%m-%d-%H%M'))
    save_roc_pr_curve_data(scores, labels, os.path.join(res_dir, res_file_name))
    gpu_q.put(gpu_to_use)


# --- 4. BENCHMARK: One-Class SVM ---
def _train_ocsvm_and_score(params, xtrain, test_labels, xtest):
    return roc_auc_score(test_labels, OneClassSVM(**params).fit(xtrain).decision_function(xtest))


def _raw_ocsvm_experiment(dataset_load_fn, dataset_name, single_class_ind):
    (x_train, y_train), (x_test, y_test) = dataset_load_fn()
    x_train = x_train.reshape((len(x_train), -1))
    x_test = x_test.reshape((len(x_test), -1))
    x_train_task = x_train[y_train.flatten() == single_class_ind]

    # Limit SVM training data to avoid hanging
    if dataset_name in ['cats-vs-dogs', 'creditcard', 'ieee-fraud']:
        n_samples = min(len(x_train_task), 10000)
        subsample_inds = np.random.choice(len(x_train_task), n_samples, replace=False)
        x_train_task = x_train_task[subsample_inds]
        print(f"Subsampling SVM training data to {n_samples} samples.")

    pg = ParameterGrid({'nu': np.linspace(0.1, 0.9, num=9),
                        'gamma': np.logspace(-7, 2, num=10, base=2)})

    results = Parallel(n_jobs=6)(
        delayed(_train_ocsvm_and_score)(d, x_train_task, y_test.flatten() == single_class_ind, x_test)
        for d in pg)

    best_params, _ = max(zip(pg, results), key=lambda t: t[-1])
    best_ocsvm = OneClassSVM(**best_params).fit(x_train_task)
    scores = best_ocsvm.decision_function(x_test)
    labels = y_test.flatten() == single_class_ind

    res_dir = os.path.join(RESULTS_DIR, dataset_name)
    if not os.path.exists(res_dir): os.makedirs(res_dir)
    res_file_name = '{}_raw-oc-svm_{}_{}.npz'.format(dataset_name,
                                                     get_class_name_from_index(single_class_ind, dataset_name),
                                                     datetime.now().strftime('%Y-%m-%d-%H%M'))
    save_roc_pr_curve_data(scores, labels, os.path.join(res_dir, res_file_name))


# --- Image-Only Benchmarks ---
def _cae_ocsvm_experiment(dataset_load_fn, dataset_name, single_class_ind, gpu_q):
    pass


def _dsebm_experiment(dataset_load_fn, dataset_name, single_class_ind, gpu_q):
    pass


def _dagmm_experiment(dataset_load_fn, dataset_name, single_class_ind, gpu_q):
    pass


def _adgan_experiment(dataset_load_fn, dataset_name, single_class_ind, gpu_q):
    pass


# --- MAIN RUNNER ---
def run_experiments(load_dataset_fn, dataset_name, q, n_classes):
    # Benchmarks (SVM, IF, AE) sind schon fertig -> Auskommentiert!
    # for c in range(n_classes):
    #     _raw_ocsvm_experiment(load_dataset_fn, dataset_name, c)
    #     _isolation_forest_experiment(load_dataset_fn, dataset_name, c)

    # --- OUR METHOD (GeoTrans / TabTrans) ---
    # WICHTIG: Kein Multiprocessing mehr ("Process"), sondern direkter Aufruf!
    n_runs = 1  # Nur 1 Run reicht für den Anfang
    for i in range(n_runs):
        print(f"--- RUN {i + 1}/{n_runs} ---")
        for c in range(n_classes):
            _transformations_experiment(load_dataset_fn, dataset_name, c, q)


def create_auc_table(metric='roc_auc'):
    file_path = glob(os.path.join(RESULTS_DIR, '*', '*.npz'))
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    methods = set()
    for p in file_path:
        _, f_name = os.path.split(p)
        parts = f_name.split(sep='_')
        # Robustere Namenserkennung (falls Zeitstempel Unterstriche hat)
        dataset_name = parts[0]
        method = parts[1]
        single_class_name = parts[2]

        methods.add(method)
        npz = np.load(p)
        roc_auc = npz[metric]
        results[dataset_name][single_class_name][method].append(roc_auc)

    for ds_name in results:
        for sc_name in results[ds_name]:
            for method_name in results[ds_name][sc_name]:
                roc_aucs = results[ds_name][sc_name][method_name]
                results[ds_name][sc_name][method_name] = [np.mean(roc_aucs),
                                                          0 if len(roc_aucs) == 1 else scipy.stats.sem(
                                                              np.array(roc_aucs))]

    with open('results-{}.csv'.format(metric), 'w') as csvfile:
        fieldnames = ['dataset', 'single class name'] + sorted(list(methods))
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for ds_name in sorted(results.keys()):
            for sc_name in sorted(results[ds_name].keys()):
                row_dict = {'dataset': ds_name, 'single class name': sc_name}
                row_dict.update({method_name: '{:.3f} ({:.3f})'.format(*results[ds_name][sc_name][method_name])
                                 for method_name in results[ds_name][sc_name]})
                writer.writerow(row_dict)


if __name__ == '__main__':
    # Manager/Queue lassen wir drin stehen, damit wir oben nichts ändern müssen,
    # aber wir nutzen es faktisch nicht mehr für Prozesse.
    freeze_support()
    man = Manager()
    q = man.Queue(1)
    q.put("0")

    experiments_list = [
        (load_ieee_fraud, 'ieee-fraud', 2),
    ]

    for data_load_fn, dataset_name, n_classes in experiments_list:
        run_experiments(data_load_fn, dataset_name, q, n_classes)

    create_auc_table()
    print("Done! Results saved in 'results-roc_auc.csv'.")
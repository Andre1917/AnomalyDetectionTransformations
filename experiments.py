import os
import csv
from collections import defaultdict
from glob import glob
from datetime import datetime
from multiprocessing import Manager, freeze_support
import numpy as np
import scipy.stats
from scipy.special import psi, polygamma
from sklearn.metrics import roc_auc_score
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import ParameterGrid
from sklearn.externals.joblib import Parallel, delayed
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.callbacks import ReduceLROnPlateau
import keras.backend as K

# --- Custom Modules ---
from utils import load_cifar10, load_cats_vs_dogs, load_fashion_mnist, load_cifar100, load_creditcard, load_ieee_fraud
from utils import save_roc_pr_curve_data, get_class_name_from_index
from transformations import Transformer, TabularTransformer, TabularTransformerIEEE

# Modelle
from models.wide_residual_network import create_wide_residual_network
from models.tabular_resnet import create_robust_resnet
from models.mlp import create_mlp
from models import dsebm, dagmm, adgan

RESULTS_DIR = 'results'


# --- 1. OUR METHOD (GeoTrans / TabTrans) ---
def _transformations_experiment(dataset_load_fn, dataset_name, single_class_ind, gpu_q=None):
    print(f"[{datetime.now()}] --- Starting Robust ResNet Experiment for Class {single_class_ind} ---")

    (x_train, y_train), (x_test, y_test) = dataset_load_fn()

    is_tabular = dataset_name in ['creditcard', 'ieee-fraud']

    if is_tabular:
        # Nutzung des ausgelagerten ResNets für Tabellendaten
        transformer = TabularTransformer(n_transforms=8)
        mdl = create_robust_resnet(input_dim=x_train.shape[1], n_classes=transformer.n_transforms)
    else:
        # Für Bilder bleibt das Wide ResNet
        transformer = Transformer(8, 8)
        mdl = create_wide_residual_network(x_train.shape[1:], transformer.n_transforms, 10, 4)

    mdl.compile('adam', 'categorical_crossentropy', ['acc'])

    x_train_task = x_train[y_train.flatten() == single_class_ind]

    # --- Generator (Verhindert OOM Absturz bei großen Daten) ---
    def batch_generator(x_data, batch_size, n_transforms):
        n_samples = len(x_data)
        indices = np.arange(n_samples)
        while True:
            np.random.shuffle(indices)
            for start_idx in range(0, n_samples, batch_size):
                batch_indices = indices[start_idx: start_idx + batch_size]
                x_batch = x_data[batch_indices]

                # Transformation "Just-in-Time"
                x_batch_repeated = np.repeat(x_batch, n_transforms, axis=0)
                t_inds = np.tile(np.arange(n_transforms), len(x_batch))
                x_batch_transformed = transformer.transform_batch(x_batch_repeated, t_inds)

                yield x_batch_transformed, to_categorical(t_inds, num_classes=n_transforms)

    # --- Training Parameter (Optimiert) ---
    batch_size = 128
    steps_per_epoch = len(x_train_task) // batch_size
    epochs = 40  # Erhöht für bessere Konvergenz auf 'Normal'-Daten

    # Callback: Reduziert Lernrate, wenn Plateau erreicht (für Feinschliff)
    lr_scheduler = ReduceLROnPlateau(monitor='acc', factor=0.5, patience=3, verbose=1, min_lr=1e-6)

    print(f"[{datetime.now()}] Starting Training (ResNet, {epochs} Epochs)...")

    mdl.fit_generator(
        generator=batch_generator(x_train_task, batch_size, transformer.n_transforms),
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=[lr_scheduler],
        verbose=1
    )

    # --- Normality Scoring Logic ---
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
    eval_batch_size = 512  # Großer Batch für schnelle Inferenz auf GPU

    for t_ind in range(transformer.n_transforms):
        print(f" - Evaluating Transform {t_ind + 1}/{transformer.n_transforms}")

        # Train stats
        obs_dir = mdl.predict(transformer.transform_batch(x_train_task, [t_ind] * len(x_train_task)),
                              batch_size=eval_batch_size)
        obs_dir = np.clip(obs_dir, EPSILON, 1 - EPSILON)

        alpha_0 = obs_dir.mean(axis=0) * calc_approx_alpha_sum(obs_dir)
        mle_alpha_t = fixed_point_dirichlet_mle(alpha_0, np.log(obs_dir).mean(axis=0))

        # Test predict
        test_p = mdl.predict(transformer.transform_batch(x_test, [t_ind] * len(x_test)),
                             batch_size=eval_batch_size)
        test_p = np.clip(test_p, EPSILON, 1 - EPSILON)

        scores += dirichlet_normality_score(mle_alpha_t, test_p)

    scores /= transformer.n_transforms
    labels = y_test.flatten() == single_class_ind

    if not np.all(np.isfinite(scores)):
        print(f"WARNING: NaNs found. Fixing...")
        finite_scores = scores[np.isfinite(scores)]
        min_val = np.min(finite_scores) if len(finite_scores) > 0 else 0.0
        scores[~np.isfinite(scores)] = min_val

    res_dir = os.path.join(RESULTS_DIR, dataset_name)
    if not os.path.exists(res_dir): os.makedirs(res_dir)

    # Dateiname enthält jetzt Timestamp, damit alte Runs nicht überschrieben werden
    res_file_name = '{}_transformations_{}_{}.npz'.format(dataset_name,
                                                          get_class_name_from_index(single_class_ind, dataset_name),
                                                          datetime.now().strftime('%Y-%m-%d-%H%M%S'))
    save_roc_pr_curve_data(scores, labels, os.path.join(res_dir, res_file_name))
    mdl.save_weights(os.path.join(res_dir, res_file_name.replace('.npz', '_weights.h5')))
    print(f"[{datetime.now()}] Saved result: {res_file_name}")

# 1b. IEEE Spezieller Transformations-Experiment (Aggressiver)
def _transformations_experimentIEEE(dataset_load_fn, dataset_name, single_class_ind, gpu_q=None):
    print(f"[{datetime.now()}] --- Starting IEEE SPECIAL Transformations (Aggressive) for Class {single_class_ind} ---")

    (x_train, y_train), (x_test, y_test) = dataset_load_fn()

    # HIER IST DER UNTERSCHIED:
    transformer = TabularTransformerIEEE(n_transforms=8)  # Aggressiver Transformer

    mdl = create_mlp(input_dim=x_train.shape[1], n_classes=transformer.n_transforms)  # Einfaches Modell

    mdl.compile('adam', 'categorical_crossentropy', ['acc'])
    x_train_task = x_train[y_train.flatten() == single_class_ind]

    # Exakt gleicher Generator & Scoring Code wie oben, nur nutzt er "transformer" (welcher jetzt der IEEE ist)
    def batch_generator(x_data, batch_size, n_transforms):
        n_samples = len(x_data)
        indices = np.arange(n_samples)
        while True:
            np.random.shuffle(indices)
            for start_idx in range(0, n_samples, batch_size):
                batch_indices = indices[start_idx: start_idx + batch_size]
                x_batch = x_data[batch_indices]
                x_batch_repeated = np.repeat(x_batch, n_transforms, axis=0)
                t_inds = np.tile(np.arange(n_transforms), len(x_batch))
                x_batch_transformed = transformer.transform_batch(x_batch_repeated, t_inds)
                yield x_batch_transformed, to_categorical(t_inds, num_classes=n_transforms)

    mdl.fit_generator(generator=batch_generator(x_train_task, 64, transformer.n_transforms),
                      steps_per_epoch=len(x_train_task) // 64, epochs=20, verbose=1)

    # --- Scoring Logic (Copy & Paste vom Standard) ---
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
        for _ in range(iters): x = x - (psi(x) - y) / polygamma(1, x)
        return x

    def fixed_point_dirichlet_mle(alpha_init, log_p_hat, max_iter=1000):
        alpha_new = alpha_old = alpha_init
        for _ in range(max_iter):
            alpha_new = inv_psi(psi(np.sum(alpha_old)) + log_p_hat)
            if np.sqrt(np.sum((alpha_old - alpha_new) ** 2)) < 1e-9: break
            alpha_old = alpha_new
        return alpha_new

    def dirichlet_normality_score(alpha, p):
        return np.sum((alpha - 1) * np.log(np.clip(p, EPSILON, 1 - EPSILON)), axis=-1)

    print(f"[{datetime.now()}] Calculating Scores (IEEE Special)...")
    scores = np.zeros((len(x_test),))
    eval_batch_size = 512

    for t_ind in range(transformer.n_transforms):
        print(f" - Transform {t_ind + 1}/{transformer.n_transforms}")
        obs_dir = mdl.predict(transformer.transform_batch(x_train_task, [t_ind] * len(x_train_task)),
                              batch_size=eval_batch_size)
        obs_dir = np.clip(obs_dir, EPSILON, 1 - EPSILON)
        alpha_0 = obs_dir.mean(axis=0) * calc_approx_alpha_sum(obs_dir)
        mle_alpha_t = fixed_point_dirichlet_mle(alpha_0, np.log(obs_dir).mean(axis=0))
        test_p = mdl.predict(transformer.transform_batch(x_test, [t_ind] * len(x_test)), batch_size=eval_batch_size)
        scores += dirichlet_normality_score(mle_alpha_t, np.clip(test_p, EPSILON, 1 - EPSILON))

    scores /= transformer.n_transforms
    labels = y_test.flatten() == single_class_ind

    if not np.all(np.isfinite(scores)): scores[~np.isfinite(scores)] = np.min(scores[np.isfinite(scores)]) if np.any(
        np.isfinite(scores)) else 0.0

    # Speichern mit speziellem Namen, damit man es in der Tabelle erkennt
    res_dir = os.path.join(RESULTS_DIR, dataset_name)
    if not os.path.exists(res_dir): os.makedirs(res_dir)
    # HIER: Name ändern auf 'transformationsIEEE'
    res_file_name = '{}_transformationsIEEE_{}_{}.npz'.format(dataset_name,
                                                              get_class_name_from_index(single_class_ind, dataset_name),
                                                              datetime.now().strftime('%Y-%m-%d-%H%M'))
    save_roc_pr_curve_data(scores, labels, os.path.join(res_dir, res_file_name))
    mdl.save_weights(os.path.join(res_dir, res_file_name.replace('.npz', '_weights.h5')))
    print(f"[{datetime.now()}] Saved IEEE Special result: {res_file_name}")

# --- 2. BENCHMARK: Isolation Forest ---
def _train_if_and_score(params, xtrain, test_labels, xtest):
    # Single-Threaded execution
    clf = IsolationForest(**params, n_jobs=1, random_state=42).fit(xtrain)
    return roc_auc_score(test_labels, -clf.decision_function(xtest))


def _isolation_forest_experiment(dataset_load_fn, dataset_name, single_class_ind):
    print(f"[{datetime.now()}] Starting Isolation Forest Experiment for Class {single_class_ind}...")
    (x_train, y_train), (x_test, y_test) = dataset_load_fn()
    x_train = x_train.reshape((len(x_train), -1))
    x_test = x_test.reshape((len(x_test), -1))
    x_train_task = x_train[y_train.flatten() == single_class_ind]

    # FAIRNESS: Bis zu 250k Samples nutzen
    if len(x_train_task) > 250000 and dataset_name == 'ieee-fraud':
        inds = np.random.choice(len(x_train_task), 250000, replace=False)
        x_train_task = x_train_task[inds]
        print(f"Subsampled IF to 250k samples (Fairness Upgrade).")

    pg = ParameterGrid({'contamination': [0.01, 0.05, 0.1], 'n_estimators': [100]})
    results = Parallel(n_jobs=1)(
        delayed(_train_if_and_score)(d, x_train_task, y_test.flatten() == single_class_ind, x_test)
        for d in pg)

    best_params, _ = max(zip(pg, results), key=lambda t: t[-1])
    print(f"Best IF Params: {best_params}")

    best_clf = IsolationForest(**best_params, n_jobs=1, random_state=42).fit(x_train_task)
    scores = -best_clf.decision_function(x_test)
    labels = y_test.flatten() == single_class_ind

    res_dir = os.path.join(RESULTS_DIR, dataset_name)
    if not os.path.exists(res_dir): os.makedirs(res_dir)
    fn = '{}_isolation-forest_{}_{}.npz'.format(dataset_name, get_class_name_from_index(single_class_ind, dataset_name),
                                                datetime.now().strftime('%Y-%m-%d-%H%M'))
    save_roc_pr_curve_data(scores, labels, os.path.join(res_dir, fn))
    print(f"Saved IF result: {fn}")


# --- 3. BENCHMARK: Autoencoder ---
def _tabular_autoencoder_experiment(dataset_load_fn, dataset_name, single_class_ind, gpu_q=None):
    print(f"[{datetime.now()}] Starting Autoencoder Experiment for Class {single_class_ind} (GPU Powered)...")

    (x_train, y_train), (x_test, y_test) = dataset_load_fn()
    x_train_task = x_train[y_train.flatten() == single_class_ind]

    input_dim = x_train.shape[1]
    encoding_dim = int(input_dim / 2)

    model = Sequential()
    model.add(Dense(encoding_dim, activation="relu", input_shape=(input_dim,)))
    model.add(Dense(input_dim, activation="linear"))
    model.compile(optimizer='adam', loss='mse')

    # FAIRNESS: 20 Epochen, Batch 256
    model.fit(x_train_task, x_train_task, epochs=20, batch_size=256, verbose=1)

    reconstructions = model.predict(x_test, batch_size=512)
    scores = np.mean(np.power(x_test - reconstructions, 2), axis=1)
    labels = y_test.flatten() == single_class_ind

    res_dir = os.path.join(RESULTS_DIR, dataset_name)
    if not os.path.exists(res_dir): os.makedirs(res_dir)
    fn = '{}_autoencoder_{}_{}.npz'.format(dataset_name, get_class_name_from_index(single_class_ind, dataset_name),
                                           datetime.now().strftime('%Y-%m-%d-%H%M'))
    save_roc_pr_curve_data(scores, labels, os.path.join(res_dir, fn))
    print(f"Saved AE result: {fn}")


# --- 4. BENCHMARK: One-Class SVM ---
def _train_ocsvm_and_score(params, xtrain, test_labels, xtest):
    return roc_auc_score(test_labels, OneClassSVM(**params).fit(xtrain).decision_function(xtest))


def _raw_ocsvm_experiment(dataset_load_fn, dataset_name, single_class_ind):
    print(f"[{datetime.now()}] Starting One-Class SVM Experiment for Class {single_class_ind}...")
    (x_train, y_train), (x_test, y_test) = dataset_load_fn()
    x_train = x_train.reshape((len(x_train), -1))
    x_test = x_test.reshape((len(x_test), -1))
    x_train_task = x_train[y_train.flatten() == single_class_ind]

    # FAIRNESS: 25k Samples
    if dataset_name in ['cats-vs-dogs', 'creditcard', 'ieee-fraud']:
        n_samples = min(len(x_train_task), 25000)
        inds = np.random.choice(len(x_train_task), n_samples, replace=False)
        x_train_task = x_train_task[inds]
        print(f"Subsampled SVM to {n_samples} samples.")

    # FIX: Numerische Gamma-Werte (kein 'scale') gegen Crash
    pg = ParameterGrid({
        'nu': np.linspace(0.1, 0.9, num=5),
        'gamma': np.logspace(-7, 2, num=5, base=2)
    })

    results = Parallel(n_jobs=1)(
        delayed(_train_ocsvm_and_score)(d, x_train_task, y_test.flatten() == single_class_ind, x_test)
        for d in pg)

    best_params, _ = max(zip(pg, results), key=lambda t: t[-1])
    print(f"Best SVM Params: {best_params}")

    best_ocsvm = OneClassSVM(**best_params).fit(x_train_task)
    scores = best_ocsvm.decision_function(x_test)
    labels = y_test.flatten() == single_class_ind

    res_dir = os.path.join(RESULTS_DIR, dataset_name)
    if not os.path.exists(res_dir): os.makedirs(res_dir)
    fn = '{}_raw-oc-svm_{}_{}.npz'.format(dataset_name, get_class_name_from_index(single_class_ind, dataset_name),
                                          datetime.now().strftime('%Y-%m-%d-%H%M'))
    save_roc_pr_curve_data(scores, labels, os.path.join(res_dir, fn))
    print(f"Saved SVM result: {fn}")


# --- Image-Only Benchmarks (Placeholders) ---
def _cae_ocsvm_experiment(dataset_load_fn, dataset_name, single_class_ind, gpu_q): pass


def _dsebm_experiment(dataset_load_fn, dataset_name, single_class_ind, gpu_q): pass


def _dagmm_experiment(dataset_load_fn, dataset_name, single_class_ind, gpu_q): pass


def _adgan_experiment(dataset_load_fn, dataset_name, single_class_ind, gpu_q): pass


# --- MAIN RUNNER ---
def run_experiments(load_dataset_fn, dataset_name, q, n_classes):
    print(f"=== PROCESSING DATASET: {dataset_name} ===")

    # 1. Benchmarks (AUSKOMMENTIERT, da schon berechnet)
    for c in range(n_classes):
        # _tabular_autoencoder_experiment(load_dataset_fn, dataset_name, c)
        # _isolation_forest_experiment(load_dataset_fn, dataset_name, c)
        # _raw_ocsvm_experiment(load_dataset_fn, dataset_name, c)
        pass

    # 2. OUR METHOD (Transformations) - Jetzt 5 Runs!
    n_runs = 5
    for i in range(n_runs):
        for c in range(n_classes):
            if dataset_name == 'ieee-fraud':
                # Nutze die SPEZIAL-Funktion für IEEE
                _transformations_experimentIEEE(load_dataset_fn, dataset_name, c, q)
            else:
                # Nutze die STANDARD-Funktion für alles andere (Creditcard etc.)
                _transformations_experiment(load_dataset_fn, dataset_name, c, q)


def create_auc_table(metric='roc_auc'):
    file_path = glob(os.path.join(RESULTS_DIR, '*', '*.npz'))
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    methods = set()
    for p in file_path:
        _, f_name = os.path.split(p)
        parts = f_name.split(sep='_')
        dataset_name = parts[0]
        method = parts[1]
        single_class_name = parts[2]

        methods.add(method)
        npz = np.load(p)
        roc_auc = npz[metric]
        results[dataset_name][single_class_name][method].append(roc_auc)

    with open('results-{}.csv'.format(metric), 'w') as csvfile:
        fieldnames = ['dataset', 'single class name'] + sorted(list(methods))
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for ds_name in sorted(results.keys()):
            for sc_name in sorted(results[ds_name].keys()):
                row_dict = {'dataset': ds_name, 'single class name': sc_name}

                # FIX: Berechnung direkt in der Schleife für jede Methode einzeln
                for method_name, vals in results[ds_name][sc_name].items():
                    mean_val = np.mean(vals)
                    sem_val = 0 if len(vals) == 1 else scipy.stats.sem(vals)
                    row_dict[method_name] = '{:.3f} ({:.3f})'.format(mean_val, sem_val)

                writer.writerow(row_dict)


if __name__ == '__main__':
    freeze_support()
    man = Manager()
    q = man.Queue(1);
    q.put("0")

    experiments_list = [
        (load_ieee_fraud, 'ieee-fraud', 2),
        (load_creditcard, 'creditcard', 2),
    ]

    for data_load_fn, dataset_name, n_classes in experiments_list:
        run_experiments(data_load_fn, dataset_name, q, n_classes)

    create_auc_table()
    print("Done! All results saved in 'results-roc_auc.csv'.")
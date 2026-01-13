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
from keras.optimizers import Adam

# --- Custom Modules ---
from utils import load_cifar10, load_cats_vs_dogs, load_fashion_mnist, load_cifar100, load_creditcard, load_ieee_fraud, load_elliptic
from utils import save_roc_pr_curve_data, get_class_name_from_index
from transformations import Transformer, TabularTransformer, TabularTransformerIEEE

# Modelle
from models.wide_residual_network import create_wide_residual_network
from models.tabular_resnet import create_robust_resnet
from models.mlp import create_mlp, create_wide_mlp
from models import dsebm, dagmm, adgan

# experiments.py
# Compatible with:
# - Python 3.6
# - TensorFlow 1.8.0
# - Keras 2.2.0
# - scikit-learn 0.19.1
#
# Notes:
# - CPU-first setup (CUDA disabled).
# - Main focus: transformation-based anomaly detection with stable scoring.
# - IMPORTANT: save_roc_pr_curve_data expects labels with 1 == positive class.
#              In anomaly detection, we set: 1 == anomaly, 0 == normal.

import os
import csv
from glob import glob
from datetime import datetime
from collections import defaultdict
from multiprocessing import Manager, freeze_support

import numpy as np
import scipy.stats
from scipy.special import psi, polygamma

# --- Force CPU / avoid CUDA probing ---
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import keras.backend as K
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.utils import to_categorical

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler

try:
    from sklearn.preprocessing import QuantileTransformer
    HAS_QUANTILE = True
except Exception:
    HAS_QUANTILE = False

# --- Custom modules ---
from utils import (
    load_cifar10, load_cats_vs_dogs, load_fashion_mnist, load_cifar100,
    load_creditcard, load_ieee_fraud, load_elliptic,
    save_roc_pr_curve_data, get_class_name_from_index
)
from transformations import TabularTransformer, TabularTransformerIEEE, TabularTransformerElliptic
from models.mlp import create_wide_mlp

try:
    from models.mlp import create_residual_mlp_with_embedding
    HAS_EMBED_MODEL = True
except Exception:
    HAS_EMBED_MODEL = False

RESULTS_DIR = "results"
EPSILON = 1e-15

RESULTS_DIR = 'results'


def _transformations_experiment(dataset_load_fn, dataset_name, single_class_ind, gpu_q=None, contamination=0.0, preloaded_data=None):
    print(f"[{datetime.now()}] --- Starting Robust ResNet Exp: {dataset_name} | Class {single_class_ind} | Contam: {contamination} ---")

    if preloaded_data is not None:
        (x_train, y_train), (x_test, y_test) = preloaded_data
    else:
        (x_train, y_train), (x_test, y_test) = dataset_load_fn() 

    is_tabular = dataset_name in ['creditcard', 'ieee-fraud'] or 'elliptic' in dataset_name

    if is_tabular: 
        transformer = TabularTransformer(n_transforms=8)
        mdl = create_robust_resnet(input_dim=x_train.shape[1], n_classes=transformer.n_transforms) 
    else: 
        transformer = Transformer(8, 8)
        mdl = create_wide_residual_network(x_train.shape[1:], transformer.n_transforms, 10, 4) 

    mdl.compile('adam', 'categorical_crossentropy', ['acc']) 

    x_train_normal = x_train[y_train.flatten() == single_class_ind]
    
    if contamination > 0.0:
        x_train_anom = x_train[y_train.flatten() != single_class_ind]
        
        n_normal = len(x_train_normal)
        n_inject = int(n_normal * contamination)
        
        if n_inject > len(x_train_anom):
            n_inject = len(x_train_anom)
            print(f"WARNING: Requested contamination requires {int(n_normal * contamination)} samples, but only {len(x_train_anom)} available. Using max available.")
        
        idx = np.random.choice(len(x_train_anom), n_inject, replace=False)
        x_injected = x_train_anom[idx]
        
        x_train_task = np.concatenate([x_train_normal, x_injected], axis=0)
        print(f"Contaminated Training! Added {n_inject} anomalies to {n_normal} normal samples.")
    else:
        x_train_task = x_train_normal
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

    batch_size = 128
    steps_per_epoch = len(x_train_task) // batch_size 
    epochs = 40

    lr_scheduler = ReduceLROnPlateau(monitor='acc', factor=0.5, patience=3, verbose=1, min_lr=1e-6)

    print(f"[{datetime.now()}] Starting Training (ResNet, {epochs} Epochs)...") 

    mdl.fit_generator( 
        generator=batch_generator(x_train_task, batch_size, transformer.n_transforms), 
        steps_per_epoch=steps_per_epoch, 
        epochs=epochs, 
        callbacks=[lr_scheduler], 
        verbose=1 
    ) 

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
    eval_batch_size = 512

    for t_ind in range(transformer.n_transforms): 
        print(f" - Evaluating Transform {t_ind + 1}/{transformer.n_transforms}") 

        obs_dir = mdl.predict(transformer.transform_batch(x_train_task, [t_ind] * len(x_train_task)),
                              batch_size=eval_batch_size) 
        obs_dir = np.clip(obs_dir, EPSILON, 1 - EPSILON) 

        alpha_0 = obs_dir.mean(axis=0) * calc_approx_alpha_sum(obs_dir) 
        mle_alpha_t = fixed_point_dirichlet_mle(alpha_0, np.log(obs_dir).mean(axis=0)) 

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

def _transformations_experimentIEEE(dataset_load_fn, dataset_name, single_class_ind, gpu_q=None):
    print(f"[{datetime.now()}] --- Starting IEEE SPECIAL Transformations (Robust & Wide) for Class {single_class_ind} ---")
    
    (x_train, y_train), (x_test, y_test) = dataset_load_fn()
    
    transformer = TabularTransformerIEEE(n_transforms=8)
    
    from models.mlp import create_wide_mlp
    mdl = create_wide_mlp(input_dim=x_train.shape[1], n_classes=transformer.n_transforms)
    
    opt = Adam(lr=0.0005)
    mdl.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])
    
    x_train_task = x_train[y_train.flatten() == single_class_ind]
    
    def batch_generator(x_data, batch_size, n_transforms):
        n_samples = len(x_data)
        indices = np.arange(n_samples)
        while True:
            np.random.shuffle(indices)
            for start_idx in range(0, n_samples, batch_size):
                batch_indices = indices[start_idx : start_idx + batch_size]
                x_batch = x_data[batch_indices]
                
                x_batch_repeated = np.repeat(x_batch, n_transforms, axis=0)
                t_inds = np.tile(np.arange(n_transforms), len(x_batch))
                x_batch_transformed = transformer.transform_batch(x_batch_repeated, t_inds)
                
                yield x_batch_transformed, to_categorical(t_inds, num_classes=n_transforms)

    batch_size = 64
    epochs = 30
    
    lr_reducer = ReduceLROnPlateau(monitor='acc', factor=0.5, patience=3, verbose=1, min_lr=1e-6)
    
    print(f"[{datetime.now()}] Training Robust Wide MLP for {epochs} epochs...")
    
    mdl.fit_generator(
        generator=batch_generator(x_train_task, batch_size, transformer.n_transforms),
        steps_per_epoch=len(x_train_task)//batch_size, 
        epochs=epochs, 
        callbacks=[lr_reducer],
        verbose=1
    )

    EPSILON = 1e-15
    def calc_approx_alpha_sum(observations):
        N = len(observations)
        f = np.mean(observations, axis=0)
        f = np.clip(f, EPSILON, 1 - EPSILON)
        obs_safe = np.clip(observations, EPSILON, 1 - EPSILON)
        return (N * (len(f) - 1) * (-psi(1))) / (N * np.sum(f * np.log(f)) - np.sum(f * np.sum(np.log(obs_safe), axis=0)))
    
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
        print(f" - Transform {t_ind+1}/{transformer.n_transforms}")
        obs_dir = mdl.predict(transformer.transform_batch(x_train_task, [t_ind] * len(x_train_task)), batch_size=eval_batch_size)
        obs_dir = np.clip(obs_dir, EPSILON, 1 - EPSILON)
        alpha_0 = obs_dir.mean(axis=0) * calc_approx_alpha_sum(obs_dir)
        mle_alpha_t = fixed_point_dirichlet_mle(alpha_0, np.log(obs_dir).mean(axis=0))
        test_p = mdl.predict(transformer.transform_batch(x_test, [t_ind] * len(x_test)), batch_size=eval_batch_size)
        scores += dirichlet_normality_score(mle_alpha_t, np.clip(test_p, EPSILON, 1 - EPSILON))

    scores /= transformer.n_transforms
    labels = y_test.flatten() == single_class_ind
    
    if not np.all(np.isfinite(scores)): 
        scores[~np.isfinite(scores)] = np.min(scores[np.isfinite(scores)]) if np.any(np.isfinite(scores)) else 0.0

    res_dir = os.path.join(RESULTS_DIR, dataset_name)
    if not os.path.exists(res_dir): os.makedirs(res_dir)
    res_file_name = '{}_transformationsIEEE_{}_{}.npz'.format(dataset_name, get_class_name_from_index(single_class_ind, dataset_name), datetime.now().strftime('%Y-%m-%d-%H%M'))
    save_roc_pr_curve_data(scores, labels, os.path.join(res_dir, res_file_name))
    mdl.save_weights(os.path.join(res_dir, res_file_name.replace('.npz', '_weights.h5')))
    print(f"[{datetime.now()}] Saved IEEE Special result: {res_file_name}")

def setup_tf1_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # harmless on CPU
    sess = tf.Session(config=config)
    K.set_session(sess)

def set_global_seed(seed):
    np.random.seed(seed)
    try:
        tf.set_random_seed(seed)
    except Exception:
        pass

def calc_approx_alpha_sum(observations):
    N = len(observations)
    f = np.mean(observations, axis=0)
    f = np.clip(f, EPSILON, 1 - EPSILON)
    obs_safe = np.clip(observations, EPSILON, 1 - EPSILON)
    denom = (N * np.sum(f * np.log(f)) - np.sum(f * np.sum(np.log(obs_safe), axis=0)))
    if abs(denom) < 1e-30:
        return 1.0
    return (N * (len(f) - 1) * (-psi(1))) / denom


def inv_psi(y, iters=5):
    cond = y >= -2.22
    x = cond * (np.exp(y) + 0.5) + (1 - cond) * (-1.0 / (y - psi(1)))
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

def anomaly_labels_int_from_normal_class(y_test, normal_class):
    y = y_test.flatten()
    return (y != normal_class).astype(np.int32)


def ensure_finite(scores):
    scores = np.asarray(scores).astype(np.float64)
    if np.all(np.isfinite(scores)):
        return scores
    finite = np.isfinite(scores)
    if np.any(finite):
        scores[~finite] = np.min(scores[finite])
    else:
        scores[:] = 0.0
    return scores


def make_results_dir(dataset_name):
    path = os.path.join(RESULTS_DIR, dataset_name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def save_run(dataset_name, method, normal_class_ind, scores, anomaly_labels_int, model=None, suffix=""):
    res_dir = make_results_dir(dataset_name)
    run_id = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    class_name = get_class_name_from_index(normal_class_ind, dataset_name)

    fname = "{ds}__{m}__{c}__{rid}{suf}.npz".format(
        ds=dataset_name, m=method, c=class_name, rid=run_id, suf=suffix
    )
    out_path = os.path.join(res_dir, fname)

    save_roc_pr_curve_data(scores, anomaly_labels_int, out_path)

    if model is not None:
        w_path = os.path.join(res_dir, fname.replace(".npz", "_weights.h5"))
        model.save_weights(w_path)

    print("[{}] Saved: {}".format(datetime.now(), fname))

def scale_tabular_robust(x_train, x_test):
    scaler = RobustScaler()
    return scaler.fit_transform(x_train), scaler.transform(x_test)

def scale_tabular_quantile(x_train, x_test):
    if not HAS_QUANTILE:
        return scale_tabular_robust(x_train, x_test)
    scaler = QuantileTransformer(output_distribution="normal", n_quantiles=2000, random_state=42)
    return scaler.fit_transform(x_train), scaler.transform(x_test)

def entropy_scores(model, x_test, transformer, batch_size=2048, mode="max", eps=1e-12):
    n = len(x_test)
    if mode not in ["mean", "max"]:
        mode = "max"

    if mode == "mean":
        scores = np.zeros((n,), dtype=np.float32)
    else:
        scores = np.full((n,), -1e9, dtype=np.float32)

    for t_ind in range(transformer.n_transforms):
        x_t = transformer.transform_batch(x_test, [t_ind] * n)
        p = model.predict(x_t, batch_size=batch_size)
        p = np.clip(p, eps, 1.0)
        ent = -np.sum(p * np.log(p), axis=1).astype(np.float32)

        if mode == "mean":
            scores += ent
        else:
            scores = np.maximum(scores, ent)

    if mode == "mean":
        scores /= float(transformer.n_transforms)

    return scores


def dirichlet_scores(model, x_train_task, x_test, transformer, batch_size=512):
    n_test = len(x_test)
    scores_normal = np.zeros((n_test,), dtype=np.float64)

    for t_ind in range(transformer.n_transforms):
        obs = model.predict(
            transformer.transform_batch(x_train_task, [t_ind] * len(x_train_task)),
            batch_size=batch_size
        )
        obs = np.clip(obs, EPSILON, 1 - EPSILON)

        alpha0 = obs.mean(axis=0) * calc_approx_alpha_sum(obs)
        mle_alpha = fixed_point_dirichlet_mle(alpha0, np.log(obs).mean(axis=0))

        test_p = model.predict(
            transformer.transform_batch(x_test, [t_ind] * n_test),
            batch_size=batch_size
        )
        test_p = np.clip(test_p, EPSILON, 1 - EPSILON)
        scores_normal += dirichlet_normality_score(mle_alpha, test_p)

    scores_normal /= float(transformer.n_transforms)
    return (-scores_normal).astype(np.float64)


def build_transform_model(input_dim, n_transforms, lr=1e-3, clipnorm=1.0):
    model = create_wide_mlp(input_dim=input_dim, n_classes=n_transforms)
    opt = Adam(lr=lr, clipnorm=clipnorm)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["acc"])
    return model


def batch_generator_transform(x_data, transformer, batch_size):
    n_samples = len(x_data)
    indices = np.arange(n_samples)

    while True:
        np.random.shuffle(indices)
        for start_idx in range(0, n_samples, batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            x_batch = x_data[batch_indices]

            t = transformer.n_transforms
            x_rep = np.repeat(x_batch, t, axis=0)
            t_inds = np.tile(np.arange(t), len(x_batch))

            x_trans = transformer.transform_batch(x_rep, t_inds)
            y_trans = to_categorical(t_inds, num_classes=t)
            yield x_trans, y_trans


def train_transform_model(model, x_train_task, transformer, batch_size=128, epochs=80):
    lr_sched = ReduceLROnPlateau(monitor="loss", factor=0.3, patience=3, verbose=1, min_lr=1e-6)
    early = EarlyStopping(monitor="loss", patience=8, restore_best_weights=True, verbose=1)

    steps = max(1, len(x_train_task) // batch_size)
    model.fit_generator(
        generator=batch_generator_transform(x_train_task, transformer, batch_size),
        steps_per_epoch=steps,
        epochs=epochs,
        callbacks=[lr_sched, early],
        verbose=1
    )


def _transformations_experimentElliptic(
    dataset_load_fn,
    dataset_name,
    normal_class_ind,
    seed,
    preloaded_data=None,
    scoring="entropy_max",       # "entropy_max", "entropy_mean", "dirichlet"
    use_quantile_scaling=False,  # keep False by default for elliptic/ieee
    mask_indices=None
):
    set_global_seed(seed)

    print("[{}] >>> TRANSFORMATIONS RUN: {} (normal_class={}, seed={}) <<<".format(
        datetime.now(), dataset_name, normal_class_ind, seed
    ))

    if preloaded_data is not None:
        (x_train, y_train), (x_test, y_test) = preloaded_data
    else:
        (x_train, y_train), (x_test, y_test) = dataset_load_fn()

    x_train = x_train.reshape((len(x_train), -1))
    x_test = x_test.reshape((len(x_test), -1))

    if mask_indices is not None and len(mask_indices) > 0:
        x_train = x_train.copy()
        x_test = x_test.copy()
        x_train[:, mask_indices] = 0.0
        x_test[:, mask_indices] = 0.0

    # Scaling choice (explicit)
    if use_quantile_scaling:
        x_train, x_test = scale_tabular_quantile(x_train, x_test)
    else:
        # Keep as-is by default (important for elliptic/ieee where loader already scales)
        pass

    ytr = y_train.flatten()
    x_train_task = x_train[ytr == normal_class_ind]
    print("    -> Training samples (normal class): {}".format(len(x_train_task)))

    if dataset_name.startswith("ieee"):
        transformer = TabularTransformerIEEE(n_transforms=8)
    elif dataset_name.startswith("elliptic"):
        transformer = TabularTransformerElliptic(n_transforms=8)
    else:
        transformer = TabularTransformer(n_transforms=8)

    try:
        transformer.fit_stats(x_train_task)
    except Exception:
        pass

    model = build_transform_model(
        input_dim=x_train.shape[1],
        n_transforms=transformer.n_transforms,
        lr=1e-3,
        clipnorm=1.0
    )

    train_transform_model(model, x_train_task, transformer, batch_size=128, epochs=80)

    if scoring == "entropy_mean":
        scores = entropy_scores(model, x_test, transformer, mode="mean")
        method_name = "transformations_entropy_mean"
    elif scoring == "dirichlet":
        scores = dirichlet_scores(model, x_train_task, x_test, transformer)
        method_name = "transformations_dirichlet"
    else:
        scores = entropy_scores(model, x_test, transformer, mode="max")
        method_name = "transformations_entropy_max"

    scores = ensure_finite(scores)
    anomaly_labels_int = anomaly_labels_int_from_normal_class(y_test, normal_class_ind)

    save_run(
        dataset_name=dataset_name,
        method=method_name,
        normal_class_ind=normal_class_ind,
        scores=scores,
        anomaly_labels_int=anomaly_labels_int,
        model=model,
        suffix="__seed{}".format(seed)
    )


# ---------------------------------------------------------------------
# Experiment: Transformations + Embedding + OCSVM (0.715-style pipeline)
# ---------------------------------------------------------------------
def _transformations_embedding_ocsvm_experiment(
    dataset_load_fn,
    dataset_name,
    normal_class_ind,
    seed,
    preloaded_data=None,
    mask_indices=None
):
    """
    This implements the earlier pipeline that produced filenames like:
      dataset__transformations-embed-ocsvm__ClassName__<timestamp>.npz

    Key details kept (because they can change AUC materially):
      - QuantileTransformer(output_distribution='normal') on *raw features*
      - Embedding head with emb_dim=128 and dropout=0.2
      - Optional label smoothing if available (TF1-era fallback)
      - StandardScaler on embeddings before OCSVM
      - OCSVM fixed params: nu=0.1, gamma=2**-5
    """
    if not HAS_EMBED_MODEL:
        print("[{}] Skipping transformations-embed-ocsvm (embedding model not available).".format(datetime.now()))
        return

    set_global_seed(seed)

    print("[{}] >>> TRANSFORMATIONS + EMBEDDING + OCSVM: {} (normal_class={}, seed={}) <<<".format(
        datetime.now(), dataset_name, normal_class_ind, seed
    ))

    if preloaded_data is not None:
        (x_train, y_train), (x_test, y_test) = preloaded_data
    else:
        (x_train, y_train), (x_test, y_test) = dataset_load_fn()

    x_train = x_train.reshape((len(x_train), -1))
    x_test = x_test.reshape((len(x_test), -1))

    if mask_indices is not None and len(mask_indices) > 0:
        x_train = x_train.copy()
        x_test = x_test.copy()
        x_train[:, mask_indices] = 0.0
        x_test[:, mask_indices] = 0.0

    # IMPORTANT: Quantile scaling on features (kept to match old 0.715 pipeline)
    x_train_q, x_test_q = scale_tabular_quantile(x_train, x_test)

    # Transformer (Elliptic specific)
    if dataset_name.startswith("ieee"):
        transformer = TabularTransformerIEEE(n_transforms=8)
    elif dataset_name.startswith("elliptic"):
        transformer = TabularTransformerElliptic(n_transforms=8)
    else:
        transformer = TabularTransformer(n_transforms=8)

    # Train only on normal class
    ytr = y_train.flatten()
    x_train_task = x_train_q[ytr == normal_class_ind]
    print("    -> Training on {} samples (normal class={}).".format(len(x_train_task), normal_class_ind))

    # Model with embedding head
    clf_model, emb_model = create_residual_mlp_with_embedding(
        input_dim=x_train_q.shape[1],
        n_classes=transformer.n_transforms,
        dropout_rate=0.2,
        emb_dim=128
    )

    opt = Adam(lr=0.001, clipnorm=1.0)

    # Try label smoothing if available in this environment; otherwise fallback
    try:
        loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05)
    except Exception:
        loss_fn = "categorical_crossentropy"

    clf_model.compile(optimizer=opt, loss=loss_fn, metrics=["acc"])

    # Local generator (kept identical to older snippet)
    def batch_generator(x_data, batch_size, n_transforms):
        n_samples = len(x_data)
        indices = np.arange(n_samples)
        while True:
            np.random.shuffle(indices)
            for start_idx in range(0, n_samples, batch_size):
                batch_indices = indices[start_idx: start_idx + batch_size]
                x_batch = x_data[batch_indices]
                x_rep = np.repeat(x_batch, n_transforms, axis=0)
                t_inds = np.tile(np.arange(n_transforms), len(x_batch))
                x_trans = transformer.transform_batch(x_rep, t_inds)
                y_trans = to_categorical(t_inds, num_classes=n_transforms)
                yield x_trans, y_trans

    batch_size = 128
    epochs = 80
    lr_scheduler = ReduceLROnPlateau(monitor="loss", factor=0.3, patience=3, verbose=1, min_lr=1e-6)
    early_stopping = EarlyStopping(monitor="loss", patience=8, restore_best_weights=True, verbose=1)

    clf_model.fit_generator(
        generator=batch_generator(x_train_task, batch_size, transformer.n_transforms),
        steps_per_epoch=max(1, len(x_train_task) // batch_size),
        epochs=epochs,
        callbacks=[lr_scheduler, early_stopping],
        verbose=1
    )

    # Embeddings
    print("[{}] Extracting embeddings...".format(datetime.now()))
    emb_train = emb_model.predict(x_train_task, batch_size=2048)
    emb_test = emb_model.predict(x_test_q, batch_size=2048)

    # Standardize embeddings for OCSVM
    from sklearn.preprocessing import StandardScaler
    emb_scaler = StandardScaler()
    emb_train = emb_scaler.fit_transform(emb_train)
    emb_test = emb_scaler.transform(emb_test)

    # OCSVM on embeddings
    from sklearn.svm import OneClassSVM
    ocsvm = OneClassSVM(kernel="rbf", nu=0.1, gamma=2 ** -5)
    ocsvm.fit(emb_train)

    # decision_function: higher = more normal -> invert for anomaly
    scores = -ocsvm.decision_function(emb_test)
    scores = ensure_finite(scores)

    anomaly_labels_int = anomaly_labels_int_from_normal_class(y_test, normal_class_ind)

    # IMPORTANT: method name contains hyphens to match the older file naming
    # This is what produced "transformations-embed-ocsvm" in the CSV.
    save_run(
        dataset_name=dataset_name,
        method="transformations-embed-ocsvm",
        normal_class_ind=normal_class_ind,
        scores=scores,
        anomaly_labels_int=anomaly_labels_int,
        model=clf_model,
        suffix="__seed{}".format(seed)
    )


# ---------------------------------------------------------------------
# Elliptic Ablation (Transformations-only suite)
# ---------------------------------------------------------------------
def elliptic_ablation(load_fn, normal_class_ind, seeds):
    print("[{}] Elliptic ablation study...".format(datetime.now()))
    (x_train, y_train), (x_test, y_test) = load_fn()

    ablations = {
        "elliptic_baseline": [],
        "elliptic_no_time": [0],
        "elliptic_no_local": list(range(1, 94)),
        "elliptic_no_aggregated": list(range(94, 166)),
    }

    preloaded = ((x_train, y_train), (x_test, y_test))
    for ds_name, mask_idx in ablations.items():
        for seed in seeds:
            _transformations_experiment(
                dataset_load_fn=None,
                dataset_name=ds_name,
                normal_class_ind=normal_class_ind,
                seed=seed,
                preloaded_data=preloaded,
                scoring="entropy_max",
                use_quantile_scaling=False,
                mask_indices=mask_idx
            )
            _transformations_experiment(
                dataset_load_fn=None,
                dataset_name=ds_name,
                normal_class_ind=normal_class_ind,
                seed=seed,
                preloaded_data=preloaded,
                scoring="entropy_mean",
                use_quantile_scaling=False,
                mask_indices=mask_idx
            )
            _transformations_embedding_ocsvm_experiment(
                dataset_load_fn=None,
                dataset_name=ds_name,
                normal_class_ind=normal_class_ind,
                seed=seed,
                preloaded_data=preloaded,
                mask_indices=mask_idx
            )

if __name__ == "__main__":
    freeze_support()
    setup_tf1_session()

    man = Manager()
    q = man.Queue(1)
    q.put("0")

    seeds = [0, 1, 2, 3, 4]

    # Primary target: Elliptic (normal = licit = 0)
    run_scientific_suite(load_elliptic, "elliptic", normal_class_ind=0, seeds=seeds, run_dirichlet=False)

    # Optional: Elliptic ablation (also runs embed+ocsvm per setting/seed)
    # elliptic_ablation(load_elliptic, normal_class_ind=0, seeds=seeds)

    create_auc_table(metric="roc_auc")
    print("Done. Results saved in '{}' and CSV generated.".format(RESULTS_DIR))
    
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

    # 1. Benchmarks (optional auskommentiert lassen)
    for c in range(n_classes):
        _tabular_autoencoder_experiment(load_dataset_fn, dataset_name, c)
    #   _isolation_forest_experiment(load_dataset_fn, dataset_name, c)
    #     _raw_ocsvm_experiment(load_dataset_fn, dataset_name, c)

    # 2. ELLIPTIC: Spezielle Logik (Optimized Ablation)
    # Dieser Datensatz nutzt nicht die Standard-Loops für Contamination/Klassen.
    # if dataset_name == 'elliptic':
        # Hier rufen wir jetzt die NEUE, optimierte Methode auf (Aggressive Trans + MLP)
        # _ablation_experiment_elliptic_optimized(load_dataset_fn, q)
        # return  # Funktion beenden, da Elliptic hier fertig ist.

    # 3. STANDARD & CONTAMINATION EXPERIMENTS (CreditCard, IEEE, etc.)
    # Wir testen: Clean (0%), 1% Contamination, 5% Contamination
    contamination_levels = [0.0, 0.01, 0.05]
    
    # Anzahl der Wiederholungen (3 Runs für Statistik)
    n_runs = 3
    
    for i in range(n_runs):
        print(f"--- Run {i+1}/{n_runs} ---")
        for c in range(n_classes):
            for contam in contamination_levels:
                
                # Fall A: IEEE-CIS Fraud Detection
                if dataset_name == 'ieee-fraud':
                    # IEEE nutzen wir nur mit der 'Special' Methode ohne Contamination (Clean Baseline)
                    # Da die Architektur dort sehr spezifisch ist (Wide MLP), mischen wir hier keine Contamination dazu.
                    if contam == 0.0:
                        _transformations_experimentIEEE(load_dataset_fn, dataset_name, c, q)
                
                # Fall B: Credit Card Fraud (und andere Standard-Sets)
                else:
                    # Hier nutzen wir die Standard-Funktion, die jetzt den 'contamination'-Parameter akzeptiert
                    _transformations_experiment(load_dataset_fn, dataset_name, c, q, contamination=contam)


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
        # (load_ieee_fraud, 'ieee-fraud', 2), 
        #(load_creditcard, 'creditcard', 2), 
        (load_elliptic, 'elliptic', 2), # Class Count ist irrelevant da intern handled
    ] 

    for data_load_fn, dataset_name, n_classes in experiments_list: 
        run_experiments(data_load_fn, dataset_name, q, n_classes)

if __name__ == '__main__':
    freeze_support()
    N_GPUS = 2
    man = Manager()
    q = man.Queue(N_GPUS)
    for g in range(N_GPUS):
        q.put(str(g))

    experiments_list = [
        (load_cifar10, 'cifar10', 10),
        (load_cifar100, 'cifar100', 20),
        (load_fashion_mnist, 'fashion-mnist', 10),
        (load_cats_vs_dogs, 'cats-vs-dogs', 2),
        # (load_ieee_fraud, 'ieee-fraud', 2),
        # (load_creditcard, 'creditcard', 2),
        # (load_elliptic, 'elliptic', 2),
    ]

    for data_load_fn, dataset_name, n_classes in experiments_list:
        run_experiments(data_load_fn, dataset_name, q, n_classes)

    create_auc_table()
    print("Done! All results saved in 'results-roc_auc.csv'.")
import numpy as np
import os
import argparse
import pickle
import copy
import gc
import tqdm
import time

import shap
import lime
import lime.lime_tabular
from alibi.explainers import ALE, PartialDependence


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pyod.models.auto_encoder import AutoEncoder
from pyod.models.cblof import CBLOF
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.ecod import ECOD
from pyod.models.copod import COPOD
from pyod.models.loda import LODA
from pyod.models.deep_svdd import DeepSVDD
from pyod.models.gmm import GMM


def train_and_get_results(clf, X, random_state):
    n_features = X.shape[1]

    X_train, X_test = train_test_split(X, train_size=0.8, random_state=random_state)

    scl = StandardScaler().fit(X_train)
    X_train = scl.transform(X_train)
    X_test = scl.transform(X_test)

    clf.fit(X_train)

    execution_times = {}

    # ---------------------------------------------------------
    # 1. SHAP (KernelExplainer)
    # ---------------------------------------------------------
    start_time = time.time()

    shp = shap.KernelExplainer(
        clf.decision_function,
        shap.kmeans(X_train, min(50, X_train.shape[0])),
    )
    shap_values = shp(X_test).values

    execution_times["SHAP"] = time.time() - start_time

    # ---------------------------------------------------------
    # 2. LIME (Local Interpretable Model-agnostic Explanations)
    # ---------------------------------------------------------
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train,
        mode="regression",
        feature_names=[str(i) for i in range(X_train.shape[1])],
        verbose=False,
        random_state=random_state,
    )

    lime_values = []

    start_time = time.time()

    for i in tqdm.tqdm(range(X_test.shape[0]), desc="LIME Progress"):
        exp = lime_explainer.explain_instance(
            X_test[i], clf.decision_function, num_features=n_features
        )

        local_exp = dict(exp.local_exp[1])
        sorted_weights = [local_exp.get(k, 0.0) for k in range(n_features)]
        lime_values.append(sorted_weights)

    lime_values = np.array(lime_values)

    execution_times["LIME"] = time.time() - start_time

    # ---------------------------------------------------------
    # 3. PD (Partial Dependence)
    # ---------------------------------------------------------
    start_time = time.time()

    pd = PartialDependence(
        clf.decision_function,
        target_names=["anomaly_score"],
        feature_names=[str(i) for i in range(n_features)],
    )

    pd_exp = pd.explain(X_train, kind="average")

    pd_results = {}
    for feature_idx in tqdm.tqdm(range(n_features), desc="PD Progress"):
        pd_results[feature_idx] = {
            "grid_values": pd_exp.feature_values[feature_idx],
            "pd_values": pd_exp.pd_values[feature_idx],
        }

    execution_times["PD"] = time.time() - start_time

    # ---------------------------------------------------------
    # 4. ALE (Accumulated Local Effects)
    # ---------------------------------------------------------
    print("ALE")
    ale = ALE(
        clf.decision_function,
        feature_names=[str(i) for i in range(n_features)],
        target_names=["anomaly_score"],
    )
    start_time = time.time()

    ale_exp = ale.explain(X_train)

    ale_results = {}
    for feature_idx in tqdm.tqdm(range(n_features), desc="ALE Progress"):
        ale_results[feature_idx] = {
            "grid_values": ale_exp.feature_values[feature_idx],
            "ale_values": ale_exp.ale_values[feature_idx],
        }

    # print(ale_exp.ale_values[feature_idx])
    execution_times["ALE"] = time.time() - start_time

    return {
        "predictions": clf.predict(X_test),
        "scores": clf.decision_function(X_test),
        "shapvalues": shap_values,
        "limevalues": lime_values,
        "pdp_values": pd_results,
        "ale_values": ale_results,
        "times": execution_times,
    }


def compute_experiment(
    dataset_path,
    save_path,
    classifiers,
    n_fold=5,
    seed=0,
):
    print(f"Processing: {dataset_path}")

    # Load Dataset
    data = np.load(dataset_path, allow_pickle=True)
    X, y = data["X"], data["y"]

    # Save ground truth
    for i in range(n_fold):
        y_train, y_test = train_test_split(y, train_size=0.8, random_state=i)
        os.makedirs(f"{save_path}/ground_truth", exist_ok=True)
        with open(f"{save_path}/ground_truth/{i}.pkl", "wb") as f:
            pickle.dump(y_test, f)

    for clf_name, base_clf in classifiers.items():

        clf = copy.deepcopy(base_clf)
        print(f"  -> {clf_name}")

        os.makedirs(f"{save_path}/{clf_name}", exist_ok=True)

        if hasattr(clf, "n_features"):
            setattr(clf, "n_features", X.shape[1])
        setattr(clf, "random_state", seed)
        if hasattr(clf, "verbose"):
            setattr(clf, "verbose", 0)

        for i in range(n_fold):
            file_out = f"{save_path}/{clf_name}/{i}.pkl"

            results = train_and_get_results(clf, X, i)

            with open(file_out, "wb") as f:
                pickle.dump(results, f)

        gc.collect()


if __name__ == "__main__":

    # Parse arguments
    # parser = argparse.ArgumentParser(description="Process dataset ID.")
    # parser.add_argument("--id", type=int, required=True, help="ID of the dataset")
    # args = parser.parse_args()
    # dataset_id = args.id

    datasets = [
        # "2_annthyroid",
        # "4_breastw",
        "14_glass",
        "15_Hepatitis",
        # "21_Lymphography",
        # "23_mammography",
        # "27_PageBlocks",
        # "29_Pima",
        # "37_Stamps",
        # "38_thyroid",
        # "39_vertebral",
        # "40_vowels",
        # "42_WBC",
        # "44_Wilt",
        # "45_wine",
        # "47_yeast",
    ]

    for dataset in datasets:

        print(dataset)

        # Set random seed
        seed = 0
        np.random.seed(0)

        classifiers = {
            "AutoEncoder": AutoEncoder(),
            "CBLOF": CBLOF(),
            "HBOS": HBOS(),
            "IForest": IForest(),
            "KNN": KNN(),
            "LOF": LOF(),
            "MCD": MCD(),
            "OCSVM": OCSVM(),
            "PCA": PCA(),
            "ECOD": ECOD(),
            "COPOD": COPOD(),
            "LODA": LODA(),
            "DeepSVDD": DeepSVDD(n_features=1),
            "GMM": GMM(),
        }

        dataset_path = f"datasets/{dataset}.npz"

        save_dir = "results_all"

        os.makedirs(save_dir, exist_ok=True)

        save_path = f"{save_dir}/{dataset}"
        os.makedirs(save_path, exist_ok=True)

        compute_experiment(dataset_path, save_path, classifiers, n_fold=5, seed=seed)

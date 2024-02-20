import pandas as pd
import matplotlib.pyplot as plt
from surprise import Dataset, Reader, NormalPredictor, BaselineOnly, KNNWithZScore, SVD
import numpy as np
import random
from surprise.reader import Reader
from surprise.model_selection import cross_validate, KFold, GridSearchCV
from collections import defaultdict
from statistics import mean
from collections import Counter
import os.path as path

from surprise import SVD, accuracy

import copy
import wget
from zipfile import ZipFile


def precision_recall_at_k(predictions, k=10, threshold=3.5):
    """Return precision and recall at k metrics for each user"""

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():
        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(
            ((true_r >= threshold) and (est >= threshold))
            for (est, true_r) in user_ratings[:k]
        )

        # Precision@K: Proportion of recommended items that are relevant
        # When n_rec_k is 0, Precision is undefined. We here set it to 0.

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

        # Recall@K: Proportion of relevant items that are recommended
        # When n_rel is 0, Recall is undefined. We here set it to 0.

        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    return precisions, recalls


def set_my_folds(dataset, nfolds=5, shuffle=True):
    raw_ratings = dataset.raw_ratings
    if shuffle:
        raw_ratings = random.sample(raw_ratings, len(raw_ratings))

    chunk_size = int(1 / nfolds * len(raw_ratings))
    thresholds = [chunk_size * x for x in range(0, nfolds)]

    print("set_my_folds> len(raw_ratings): %d" % len(raw_ratings))

    folds = []

    for th in thresholds:
        test_raw_ratings = raw_ratings[th : th + chunk_size]
        train_raw_ratings = raw_ratings[:th] + raw_ratings[th + chunk_size :]

        print(
            "set_my_folds> threshold: %d, len(train_raw_ratings): %d, len(test_raw_ratings): %d"
            % (th, len(train_raw_ratings), len(test_raw_ratings))
        )

        folds.append((train_raw_ratings, test_raw_ratings))

    return folds


def get_raw_dataset(url):
    if path.exists("ml-latest-small"):
        return "ml-latest-small/ratings.csv"

    else:
        wget.download(url)
        test_file_name = "ml-latest-small.zip"

        with ZipFile(test_file_name, "r") as zip:
            zip.extractall()

        data_path = "ml-latest-small/ratings.csv"

        return data_path


def summarize_dataset_info(dataset):
    users = dataset["userId"]
    rating = dataset["rating"]
    products = dataset["movieId"]

    missing_values = dataset.isna().sum()
    duplicates = dataset.duplicated().sum()

    print("\nMising values :")
    print(missing_values)
    print("\nDuplicated values :", duplicates)

    print("Num users : ", users.nunique())
    print("Num ratings : ", rating.nunique())
    print("Num products : ", products.nunique())

    print("Dataset shape : ", dataset.shape)


def clean_dataset(dataset):
    product_counts = dataset["movieId"].value_counts()
    cleaned_df = dataset[
        dataset["movieId"].isin(product_counts[product_counts >= 10].index)
    ]

    user_counts = cleaned_df["userId"].value_counts()
    cleaned_df = cleaned_df[
        cleaned_df["userId"].isin(user_counts[user_counts >= 20].index)
    ]

    return cleaned_df


def plot_info_dataset(dataset):
    df1 = dataset.groupby("userId")["rating"].mean().reset_index(name="rating")
    df2 = dataset.groupby("movieId")["rating"].mean().reset_index(name="rating")

    plt.figure(figsize=(10, 6))
    plt.title("Puntuaciones por usuario")
    dataset["userId"].value_counts().plot(kind="hist")
    plt.xlabel("Número de puntuaciones")
    plt.ylabel("Número de Usuarios")
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.title("Puntuaciones por producto")
    dataset["movieId"].value_counts().plot(kind="hist")
    plt.xlabel("Número de Puntuaciones")
    plt.ylabel("Número de Productos")
    plt.show()

    plt.figure(figsize=(10, 6))
    print(df1)
    df1["rating"].plot(kind="hist", bins=30, color="blue", edgecolor="black")
    plt.title("Media de Puntuaciones por Usuario")
    plt.xlabel("Media de Puntuaciones")
    plt.ylabel("Número de Usuarios")
    plt.show()

    plt.figure(figsize=(10, 6))
    print(df2)
    df2["rating"].plot(kind="hist", bins=30, color="blue", edgecolor="black")
    plt.title("Media de Puntuaciones por Producto")
    plt.xlabel("Media de Puntuaciones")
    plt.ylabel("Número de Productos")
    plt.show()

    plt.figure(figsize=(10, 6))
    dataset["rating"].value_counts().sort_index().plot(
        kind="bar", color="blue", edgecolor="black"
    )
    plt.title("Distribución de las puntuaciones")
    plt.xlabel("Valores de las puntuaciones")
    plt.ylabel("Cantidad de puntuaciones")
    plt.show()


def create_surprise_object(dataset):
    min_rating = dataset["rating"].min()
    max_rating = dataset["rating"].max()
    print(min_rating)

    reader = Reader(rating_scale=(min_rating, max_rating))
    data = Dataset.load_from_df(dataset[["userId", "movieId", "rating"]], reader)

    print("El tipo del dataset es ", type(data))

    # Setting seed to make code reproducible
    my_seed = 311
    random.seed(my_seed)
    np.random.seed(my_seed)

    folds = set_my_folds(data)
    return data, folds


def validate_grid_search(dataset, folds, algorithm):
    if algorithm == "KNNWithZScorePearson":
        algo = KNNWithZScore()
        algo_param_grid = {
            "k": [25, 50, 100],
            "min_k": [1, 2, 5],
            "verbose": [False],
            "sim_options": {"name": ["pearson"], "verbose": ["false"]},
        }

        print(algo.__class__)
        mae_list = []
        params_list = []
        std_list = []

        for i, (train_ratings, test_ratings) in enumerate(folds):
            print("Fold: %d" % i)

            knn_gs = GridSearchCV(
                algo.__class__, algo_param_grid, measures=["mae"], cv=3, n_jobs=-1
            )

            # fit parameter must have a raw_ratings attribute
            train_dataset = copy.deepcopy(dataset)
            train_dataset.raw_ratings = train_ratings
            knn_gs.fit(train_dataset)

            results_df = pd.DataFrame.from_dict(knn_gs.cv_results)
            print(results_df)

            min_row = results_df.loc[results_df["mean_test_mae"].idxmin()]

            # Ahora min_row contiene la fila con el valor mínimo en 'mean_test_mae'
            min_k_value = min_row["param_k"]
            min_para_min_k_value = min_row["param_min_k"]

            std_list.append(results_df["std_test_mae"])

            params_list.append((min_k_value, min_para_min_k_value))
            print("min ", results_df["mean_test_mae"].min())

            # best MAE score
            print(
                "Grid search>\nmae=%.3f, cfg=%s"
                % (knn_gs.best_score["mae"], knn_gs.best_params["mae"])
            )

            # We can now use the algorithm that yields the best MAE
            knn_algo = knn_gs.best_estimator["mae"]

            # We train the algorithm with the whole train set
            knn_algo.fit(train_dataset.build_full_trainset())

            # test parameter must be a testset
            test_dataset = copy.deepcopy(dataset)
            test_dataset.raw_ratings = test_ratings
            test_set = test_dataset.construct_testset(raw_testset=test_ratings)

            svd_predictions = knn_algo.test(test_set)

            # Compute and print MAE
            print("Test>")
            accuracy.mae(svd_predictions, verbose=True)
            mae_list.append(
                (f"Fold {i}", accuracy.mae(svd_predictions), knn_gs.best_params["mae"])
            )

        print(mae_list)

        # Calcular la media del MAE
        mean_mae = mean(fold[1] for fold in mae_list)

        # Encontrar el valor de parámetros que más se repite
        print(params_list)

        counter = Counter(params_list)

        # Encontrar la combinación más común
        most_common_params = counter.most_common(1)[0][0]

        print(f"La mejor combinación es: {most_common_params}")

        std_deviation = np.mean(std_list)
        print(f"Standard Deviation of MAE: {std_deviation}")

        return mean_mae, most_common_params, algorithm, std_deviation

    elif algorithm == "NormalPredictor":
        most_common_params = []
        algo = NormalPredictor()
        algo_param_grid = {}

        print(algo.__class__)
        mae_list = []
        params_list = []
        std_list = []

        for i, (train_ratings, test_ratings) in enumerate(folds):
            print("Fold: %d" % i)

            knn_gs = GridSearchCV(
                algo.__class__, algo_param_grid, measures=["mae"], cv=3, n_jobs=-1
            )

            # fit parameter must have a raw_ratings attribute
            train_dataset = copy.deepcopy(dataset)
            train_dataset.raw_ratings = train_ratings
            knn_gs.fit(train_dataset)

            results_df = pd.DataFrame.from_dict(knn_gs.cv_results)
            print(results_df)

            print("min ", results_df["mean_test_mae"].min())
            std_list.append(results_df["std_test_mae"])

            # best MAE score
            print(
                "Grid search>\nmae=%.3f, cfg=%s"
                % (knn_gs.best_score["mae"], knn_gs.best_params["mae"])
            )

            # We can now use the algorithm that yields the best MAE
            knn_algo = knn_gs.best_estimator["mae"]

            # We train the algorithm with the whole train set
            knn_algo.fit(train_dataset.build_full_trainset())

            # test parameter must be a testset
            test_dataset = copy.deepcopy(dataset)
            test_dataset.raw_ratings = test_ratings
            test_set = test_dataset.construct_testset(raw_testset=test_ratings)

            svd_predictions = knn_algo.test(test_set)

            # Compute and print MAE
            print("Test>")
            accuracy.mae(svd_predictions, verbose=True)
            mae_list.append(
                (f"Fold {i}", accuracy.mae(svd_predictions), knn_gs.best_params["mae"])
            )

        # Extraer la columna de MAE
        print(mae_list)

        # Calcular la media del MAE
        mean_mae = mean(fold[1] for fold in mae_list)

        # desviación típica es la raíz cuadrada de los valores medios de las diferencias al cuadrado entre cada MAE y el MAE medio.
        std_deviation = np.mean(std_list)

        print(f"Standard Deviation of MAE: {std_deviation}")
        return mean_mae, most_common_params, algorithm, std_deviation

    elif algorithm == "SVD":
        most_common_params = []
        algo = SVD()
        algo_param_grid = {"n_factors": [25]}

        print(algo.__class__)
        mae_list = []
        params_list = []
        std_list = []

        for i, (train_ratings, test_ratings) in enumerate(folds):
            print("Fold: %d" % i)

            knn_gs = GridSearchCV(
                algo.__class__, algo_param_grid, measures=["mae"], cv=3, n_jobs=-1
            )

            # fit parameter must have a raw_ratings attribute
            train_dataset = copy.deepcopy(dataset)
            train_dataset.raw_ratings = train_ratings
            knn_gs.fit(train_dataset)

            results_df = pd.DataFrame.from_dict(knn_gs.cv_results)
            print(results_df)

            std_list.append(results_df["std_test_mae"])

            # best MAE score
            print(
                "Grid search>\nmae=%.3f, cfg=%s"
                % (knn_gs.best_score["mae"], knn_gs.best_params["mae"])
            )

            # We can now use the algorithm that yields the best MAE
            knn_algo = knn_gs.best_estimator["mae"]

            # We train the algorithm with the whole train set
            knn_algo.fit(train_dataset.build_full_trainset())

            # test parameter must be a testset
            test_dataset = copy.deepcopy(dataset)
            test_dataset.raw_ratings = test_ratings
            test_set = test_dataset.construct_testset(raw_testset=test_ratings)

            svd_predictions = knn_algo.test(test_set)

            # Compute and print MAE
            print("Test>")
            accuracy.mae(svd_predictions, verbose=True)
            mae_list.append(
                (f"Fold {i}", accuracy.mae(svd_predictions), knn_gs.best_params["mae"])
            )

        # Extraer la columna de MAE
        print(mae_list)

        # Calcular la media del MAE
        mean_mae = mean(fold[1] for fold in mae_list)

        # desviación típica es la raíz cuadrada de los valores medios de las diferencias al cuadrado entre cada MAE y el MAE medio.
        # print(f"Standard Deviation of MAE List: {np.mean(std_list)}")
        std_deviation = np.mean(std_list)

        # print(f"Standard Deviation of MAE: {std_deviation}")

        return mean_mae, most_common_params, algorithm, std_deviation


def plot_compare_result_algorithms(algorithms_result):
    algorithms = list(algorithms_result.keys())
    mae_values = list(algorithms_result.values())

    # Crear un gráfico de barras
    plt.bar(algorithms, mae_values, color="blue")

    # Agregar etiquetas y título
    plt.xlabel("Algoritmo y las mejores combinaciones")
    plt.ylabel("MAE")
    plt.title("Comparación de algoritmos basada en MAE")

    # Mostrar el gráfico
    plt.show()


def recomendation_task(surprise_dataset, folds, algorithm_list):
    for algorithm in algorithm_list:
        precision_list = []
        recall_list = []

        for item in [1, 2, 5, 10]:
            precision_list_fold = []
            recall_list_fold = []

            for i, (train_ratings, test_ratings) in enumerate(folds):
                print("Fold: %d" % i)
                train_dataset = copy.deepcopy(surprise_dataset)
                train_dataset.raw_ratings = train_ratings

                # We train the algorithm with the whole train set
                algorithm.fit(train_dataset.build_full_trainset())

                # test parameter must be a testset
                test_dataset = copy.deepcopy(surprise_dataset)
                test_dataset.raw_ratings = test_ratings
                test_set = test_dataset.construct_testset(raw_testset=test_ratings)

                predictions = algorithm.test(test_set)

                # Compute and print MAE
                print("Test>")

                precisions, recalls = precision_recall_at_k(
                    predictions, item, threshold=4
                )
                precision, recall = sum(prec for prec in precisions.values()) / len(
                    precisions
                ), sum(rec for rec in recalls.values()) / len(recalls)

                precision_list_fold.append(precision)
                recall_list_fold.append(recall)

                print(precision_list_fold)
                print(recall_list_fold)

            precision_list.append(np.mean(precision_list_fold))
            recall_list.append(np.mean(recall_list_fold))

        plt.plot(recall_list, precision_list, label=str(algorithm))

        print("recall_list: ")
        print(recall_list)
        print("recision_list: ")
        print(precision_list)

    plt.legend(loc="center")
    plt.show()


if __name__ == "__main__":
    data_path = get_raw_dataset(
        "http://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    )
    raw_df = pd.read_csv(data_path)

    cleaned_df = clean_dataset(raw_df)

    summarize_dataset_info(raw_df)
    print("\n\nAFTER CLEANING\n\n")
    summarize_dataset_info(cleaned_df)

    # Ejercicicio 3,4,5
    plot_info_dataset(cleaned_df)

    # Ejercicio 6
    surpise_object, folds = create_surprise_object(cleaned_df)

    # Ejercicio 7
    (
        mean_mae_knn,
        most_common_knn_params,
        algo_knn,
        knn_std_deviation,
    ) = validate_grid_search(surpise_object, folds, "KNNWithZScorePearson")
    # # Ejercicio 8
    (
        mean_mae_normal,
        most_common_normal_params,
        algo_normal,
        normal_std_deviation,
    ) = validate_grid_search(surpise_object, folds, "NormalPredictor")
    (
        mean_mae_svd,
        most_common_svd_params,
        algo_svd,
        svd_std_deviation,
    ) = validate_grid_search(surpise_object, folds, "SVD")

    print(
        "algorithm : ",
        algo_knn,
        "MAE : ",
        mean_mae_knn,
        "STD: ",
        knn_std_deviation,
        "best params : ",
        most_common_knn_params,
    )
    print(
        "algorithm : ",
        algo_normal,
        "MAE : ",
        mean_mae_normal,
        "STD: ",
        normal_std_deviation,
        "best params : ",
        most_common_normal_params,
    )
    print(
        "algorithm : ",
        algo_svd,
        "MAE : ",
        mean_mae_svd,
        "STD : ",
        svd_std_deviation,
        "best params : ",
        most_common_svd_params,
    )

    if most_common_svd_params == []:
        most_common_svd_params = "N_factor_25"

    results = {
        algo_knn + str(most_common_knn_params): mean_mae_knn,
        algo_normal + str(most_common_normal_params): mean_mae_normal,
        algo_svd + str(most_common_svd_params): mean_mae_svd,
    }

    print(results)

    plot_compare_result_algorithms(results)

    # Ejercicio 9
    recomendation_task(
        surpise_object,
        folds,
        [
            NormalPredictor(),
            SVD(n_factors=25),
            KNNWithZScore(
                verbose=False, k=25, k_min=1, sim_options={"name": "pearson"}
            ),
        ],
    )

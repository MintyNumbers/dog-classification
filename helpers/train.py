from os import makedirs
from typing import Optional, Union

import matplotlib.pyplot as plt
from keras.callbacks import History
from keras.losses import CategoricalCrossentropy
from keras.models import Sequential, load_model
from keras.optimizers import SGD, Adam
from numpy import argmax, ndarray
from seaborn import heatmap
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.neighbors._classification import KNeighborsClassifier

from helpers.cnn import init_new_model


def plot_confusion_matrix(
    x_train: ndarray,
    y_train: ndarray,
    x_test: ndarray,
    y_test: ndarray,
    model: Union[Sequential, Pipeline, KNeighborsClassifier],
    figure_suptitle: Optional[str] = None,
) -> None:
    f = plt.figure(figsize=(15, 5))
    if figure_suptitle:
        f.suptitle(figure_suptitle)
    for data_type, x, y in [("Train", x_train, y_train), ("Test", x_test, y_test)]:
        # Calculate Confusion Matrix
        if type(model) is Sequential:
            y_true = argmax(y, 1)
            y_pred = argmax(model.predict(x, verbose=0), 1)
        elif (type(model) is Pipeline) or (type(model) is KNeighborsClassifier):
            y_true = y
            y_pred = model.predict(x)
        else:
            print("Model type not supported")
            return

        cm = confusion_matrix(y_true, y_pred)

        # Visualize
        print(f"{data_type} Images: {x.shape[0]},\tAccuracy: {(accuracy_score(y_true, y_pred)*100):.4f}%")
        f.add_subplot(121 if data_type == "Train" else 122)
        heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix: {data_type}")
        plt.xlabel("Predicted")
        plt.ylabel("True")

    plt.show()


def train_cnn_kfold(
    x_train: ndarray,
    y_train: ndarray,
    x_test: ndarray,
    y_test: ndarray,
    epoch_per_kfold: int,
    batch_size: int,
    lr: float,
    regularization: float,
    sgd: bool,
    savedir: str,
    num_kfolds: int = 3,
) -> None:
    makedirs(savedir, exist_ok=False)
    kf = KFold(n_splits=num_kfolds, shuffle=True, random_state=0)
    for kfold, (train_indeces, val_indeces) in enumerate(kf.split(X=x_train, y=y_train), 1):
        print(f"K-Fold {kfold}")

        # Initialize model and optimizer
        model = init_new_model(regularization)
        model.compile(
            optimizer=SGD(lr) if sgd else Adam(lr), loss=CategoricalCrossentropy(), metrics=["accuracy", "mse"]
        )

        # Train model with each K-Fold
        _: History = model.fit(
            x=x_train[train_indeces],
            y=y_train[train_indeces],
            batch_size=batch_size,
            epochs=epoch_per_kfold,
            shuffle=True,
            validation_data=(x_train[val_indeces], y_train[val_indeces]),
            verbose=1,
        )

        # Save and evaluate model
        model.save(f"{savedir}/kfold-{kfold}.keras", save_format="keras")
        plot_confusion_matrix(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            model=model,
            figure_suptitle=f"K-Fold {kfold} - Train CM includes data from all {num_kfolds} folds",
        )


def train_cnn(
    x_train: ndarray,
    y_train: ndarray,
    x_test: ndarray,
    y_test: ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    regularization: float,
    sgd: bool,
    savepath: str,
    loadpath: Optional[str] = None,
) -> History:
    # Initialize model and compiler
    if loadpath:
        model: Sequential = load_model(loadpath)
    else:
        model = init_new_model(regularization)
    model.compile(optimizer=SGD(lr) if sgd else Adam(lr), loss=CategoricalCrossentropy(), metrics=["accuracy", "mse"])

    # Train and evaluate model
    history: History = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=epochs,
        shuffle=True,
        validation_data=(x_test, y_test),
        verbose=1,
    )

    # Save and evaluate model
    model.save(savepath, save_format="keras")
    plot_confusion_matrix(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, model=model)

    return history

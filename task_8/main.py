from tensorflow import keras
from keras.utils.vis_utils import plot_model
import numpy as np
import os
import time
from sklearn.model_selection import train_test_split
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

root_logdir = os.path.join(os.curdir, "my_logs")


def get_run_logdir():
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)


# read the data, and filter it where y values are <= 4
def read_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # indices
    train_indices = y_train <= 4
    test_indices = y_test <= 4

    full_train_x = x_train[train_indices]
    full_train_y = y_train[train_indices]

    x_test = x_test[test_indices]
    y_test = y_test[test_indices]

    full_train_x = full_train_x / 255.0
    x_test = x_test / 255.0

    x_train, x_val, y_train, y_val = train_test_split(
        full_train_x, full_train_y, test_size=0.09
    )

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def build_achitecture(
    n_hidden=5, n_neurons=100, initializer="he_normal", act_func="elu", lr=0.01
):
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    model = keras.models.Sequential()

    model.add(keras.layers.Flatten(input_shape=[28, 28]))

    for _ in range(n_hidden):
        model.add(
            keras.layers.Dense(
                units=n_neurons, kernel_initializer=initializer, activation=act_func
            )
        )

    model.add(
        keras.layers.Dense(10, activation="softmax", kernel_initializer=initializer)
    )

    adam_optimizer = keras.optimizers.Adam(learning_rate=lr)

    model.compile(
        optimizer=adam_optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def tune_parametres(params_distr):
    keras_classifier = keras.wrappers.scikit_learn.KerasClassifier(build_achitecture)

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = read_data()

    rnd_search_cv = GridSearchCV(
        keras_classifier, params_distr, cv=5, scoring="accuracy"
    )

    rnd_search_cv.fit(
        x_train,
        y_train,
        epochs=30,
        validation_data=(x_val, y_val),
        callbacks=[keras.callbacks.EarlyStopping(patience=10)],
    )

    best_model = rnd_search_cv.best_estimator_.model
    best_score = rnd_search_cv.best_score_

    best_model.save("./task8/best_tuning.h5")


def create_model():
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = read_data()
    model = build_achitecture()
    run_logdir = get_run_logdir()

    tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
    mdodel_cb = keras.callbacks.ModelCheckpoint(
        "./task_8/lowest_validation_loss.h5",
        monitor="val_loss",  # Quantity to monitor (e.g., validation loss)
        save_best_only=True,  # Save only the best model
        save_weights_only=False,  # Save the entire model (including architecture)
        mode="min",  # In this case, we want to minimize the monitored quantity
        verbose=1,
    )

    # The verbose parameter in the ModelCheckpoint callback controls the amount of information or
    # output displayed during the saving of model checkpoints. It determines the verbosity level of
    # the callback.

    model.fit(
        x=x_train,
        y=y_train,
        batch_size=32,
        epochs=50,
        validation_data=(x_val, y_val),
        callbacks=[mdodel_cb, tensorboard_cb],
        verbose=0,
    )

    model.save("./task_8/mnist_model.h5")

    return (x_test, y_test)


def main():
    # create_model()
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = read_data()

    # lowest_loss = keras.models.load_model("./task_8/lowest_validation_loss.h5")
    # model = keras.models.load_model("./task_8/mnist_model.h5")

    # llm means lowest loss model
    # llm_loss, llm_accuracy = lowest_loss.evaluate(x_test, y_test, verbose=0)
    # m_loss, m_accuracy = model.evaluate(x_test, y_test, verbose=0)

    # print(
    #     f"Lowest loss model:\n\tloss{llm_loss}\n\taccuracy:{llm_accuracy}\nMnist_model:\n\tloss{m_loss}\n\taccuracy:{m_accuracy}"
    # )

    # so we can see that the model with lowest loss during the training performed much better than the final model,which was created based on 50 epochs

    param_distribs = {"n_neurons": [100, 200, 300], "lr": [0.01, 0.001]}
    tune_parametres(params_distr=param_distribs)


if __name__ == "__main__":
    main()

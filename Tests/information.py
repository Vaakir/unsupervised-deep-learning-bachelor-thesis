from Visualizations.classifier import plot_confusion_matrix
from tensorflow.keras import layers, models

def classifier_test(ae, x_train, y_train, x_test, y_test):
    """ Check whether the functional dependency of y on x is preserved. """
    # Encode the space with the model (lossy)
    x_train = ae.encode(x_train)
    x_test = ae.encode(x_test)

    # Train classifier
    model = models.Sequential([
        layers.Input(x_train.shape[1:]),
        layers.Dense(100),
        layers.Dense(50),
        layers.Dense(y_train.shape[1], activation='softmax')
    ])

    model.compile(loss="mse", optimizer="adam")
    model.fit(x_train, y_train, epochs = 100, verbose = False)
    y_pred = model.predict(x_test)
    plot_confusion_matrix(y_pred, y_test)
import tensorflow as tf

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    assert x_train.shape == (50000, 32, 32, 3)
    assert x_test.shape == (10000, 32, 32, 3)
    assert y_train.shape == (50000, 1)
    assert y_test.shape == (10000, 1)

    model = tf.keras.applications.VGG16(weights=None, input_shape=(32, 32, 3), classes=10)
    print(model.summary())

    model.compile(loss="sparse_categorical_crossentropy")
    model.fit(x_train[:5, ...], y_train[:5], verbose=True)

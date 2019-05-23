from audio_processing import *

import tensorflow as tf


def classify_all_keys():
    data = load_keystroke_data()
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(29, activation='softmax'),
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    evaluate_model(model, data)


def classify_spacebar():
    data = load_keystroke_data_for_binary_classifier(classify={'space', 'a'})
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax'),
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    evaluate_model(model, data)
 

def evaluate_model(compiled_model, dataset, ratio=0.9, epochs=5):
    data, labels = dataset
    scaled_data = scale_keystroke_data(data)
    train_size = int(len(scaled_data) * ratio)
    x_train, x_test = scaled_data[:train_size], scaled_data[train_size:]
    y_train, y_test = labels[:train_size], labels[train_size:]
    model.fit(x_train, y_train, epochs=epochs)
    model.evaluate(x_test, y_test)


if __name__ == '__main__':
    classify_spacebar()

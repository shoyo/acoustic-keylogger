from audio_processing import *

import tensorflow as tf


def main():
    data, labels = load_keystroke_data()
    scaled_data = scale_keystroke_data(data)
    train_size = int(len(scaled_data) * 0.9)
    x_train, x_test = scaled_data[:train_size], scaled_data[train_size:]
    y_train, y_test = labels[:train_size], labels[train_size:]
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(29, activation='softmax'),
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=15)
    model.evaluate(x_test, y_test)


def space_or_no():
    data, labels = load_keystroke_data2()
    scaled_data = scale_keystroke_data(data)
    train_size = int(len(scaled_data) * 0.9)
    x_train, x_test = scaled_data[:train_size], scaled_data[train_size:]
    y_train, y_test = labels[:train_size], labels[train_size:]
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax'),
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5)
    model.evaluate(x_test, y_test)

   



if __name__ == '__main__':
    space_or_no()

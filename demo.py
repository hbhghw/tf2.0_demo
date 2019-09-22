import tensorflow as tf
from tensorflow import keras
import numpy as np


def build_dataset():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = np.reshape(x_train, [-1, 784])
    x_test = np.reshape(x_test, [-1, 784])

    def map_fun(x, y):
        y = tf.cast(y, tf.int64)
        x = tf.cast(x, tf.float32) / 255.
        return x, y

    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset = dataset.map(map_fun)
    dataset = dataset.repeat().shuffle(buffer_size=1024).batch(100)
    return dataset


def build_model():#simple model
    inputs = keras.layers.Input(shape=[784, ])
    x = keras.layers.Dense(256, activation='relu')(inputs)
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.Dense(units=10)(x)
    model = keras.Model(inputs=[inputs], outputs=[x])
    return model


def compute_loss(y_true, y_pred):
    return tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))

def compute_acc(y_true, y_pred):  # keras.metrics.sparse_categorical_accuracy
    return tf.reduce_mean(tf.cast(tf.equal(y_true, tf.argmax(y_pred, axis=1)), tf.float32))

def evaluate():
    model = keras.models.load_model('keras_model.h5')
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    y_pred = model.predict(np.reshape(x_test, [-1, 784]))
    y_pred = np.argmax(y_pred, axis=1)
    print(np.mean(y_pred == y_test))

opt = tf.keras.optimizers.Adam()
model = build_model()
dataset = build_dataset()


@tf.function
def train():
    step = 0
    for x, y_true in dataset:
        step += 1
        with tf.GradientTape() as tape:
            y_pred = model(x)
            loss = compute_loss(y_true, y_pred)
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        acc = compute_acc(y_true, y_pred)
        if tf.equal(step % 20, 0):
            tf.print('step:', step, 'loss:', loss, 'acc:', acc)
            #print('step:', step, 'loss:', loss.numpy(), 'acc:', acc.numpy())
        if tf.equal(step,2000):
            break
    #model.save('keras_model.h5') #problem here??


train()
model.save('keras_model.h5')
evaluate()


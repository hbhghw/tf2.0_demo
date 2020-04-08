import tensorflow as tf
from tensorflow import keras
import numpy as np


def conv2d(inputs, filters, kernel_size=3, strides=1, padding='same', bn=True, act=None):
    x = keras.layers.Conv2D(filters, kernel_size, strides, padding=padding)(inputs)
    if bn:
        x = keras.layers.BatchNormalization()(x)
    if act:
        x = keras.layers.ReLU()(x)
    return x


def upsample(inputs):
    h, w = inputs.shape[1:3]
    return tf.image.resize(inputs, (2 * h, 2 * w))


def reshape(inputs, output_shape):
    return keras.layers.Reshape(output_shape)(inputs)


def residual_block(inputs, filters):
    shortcut = inputs
    x = conv2d(inputs, filters, kernel_size=1, act=True)
    x = conv2d(x, filters * 2)
    x = x + shortcut
    x = keras.layers.ReLU()(x)
    return x


def darknet53(inputs):
    x = conv2d(inputs, 32)
    x = conv2d(x, 64, strides=2)
    filter_list = [32, 64, 128, 256, 512]
    repeats = [1, 2, 8, 8, 4]
    feats = []
    for i, (f, r) in enumerate(zip(filter_list, repeats)):
        for _ in range(r):
            x = residual_block(x, f)
        feats.append(x)
        if i < 4:
            x = conv2d(x, f * 4, strides=2)
    return feats[-3:]


def prediction_conv(feat1, feat2, feat3, n_classes=20):
    f3 = conv2d(feat3, 512, kernel_size=1)
    f3 = conv2d(f3, 1024)
    f3 = conv2d(f3, 512, 1)
    f3 = conv2d(f3, 1024)
    f3 = conv2d(f3, 512, 1)
    f3_branch = conv2d(f3, 1024)
    output3 = conv2d(f3_branch, 3 * (1 + 4 + n_classes), kernel_size=1, bn=False, act=None)
    f3 = conv2d(f3, 256, kernel_size=1)
    f3 = upsample(f3)
    f2 = tf.concat([feat2, f3], axis=-1)
    f2 = conv2d(f2, 256, 1)
    f2 = conv2d(f2, 512, 3)
    f2 = conv2d(f2, 256, 1)
    f2 = conv2d(f2, 512)
    f2 = conv2d(f2, 256, 1)
    f2_branch = conv2d(f2, 512)
    output2 = conv2d(f2_branch, 3 * (1 + 4 + n_classes), 1, bn=False, act=None)
    f2 = conv2d(f2, 128, 1)
    f2 = upsample(f2)
    f1 = tf.concat([feat1, f2], axis=-1)
    f1 = conv2d(f1, 128, 1)
    f1 = conv2d(f1, 256)
    f1 = conv2d(f1, 128, 1)
    f1 = conv2d(f1, 256)
    f1 = conv2d(f1, 128, 1)
    f1_branch = conv2d(f1, 256)
    output1 = conv2d(f1_branch, 3 * (1 + 4 + n_classes), 1, bn=False, act=None)
    output1 = reshape(output1, [-1, 1 + 4 + n_classes])
    output2 = reshape(output2, [-1, 1 + 4 + n_classes])
    output3 = reshape(output3, [-1, 1 + 4 + n_classes])
    output = tf.concat([output1, output2, output3], axis=-2)
    return output


# loss:[batch,n_anchors]
def l1_lose(y_ture, y_pred):
    return tf.reduce_sum(tf.abs(y_ture - y_pred), axis=-1)


def smooth_L1lose(y_true, y_pred):
    v = tf.abs(y_true - y_pred)
    x = tf.where(v < 1, 0.5 * v ** 2, v - 0.5)
    return tf.reduce_sum(x, axis=-1)


def softmax_cross_entropy(y_true, y_pred):
    y_pred = tf.nn.softmax(y_pred, axis=-1)
    return -tf.reduce_sum(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred), axis=-1)


def sigmoid_cross_entropy(y_true, y_pred):
    y_pred = tf.nn.sigmoid(y_pred)
    return -(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))


def focal_loss(y_true, y_pred, alpha=0.5, gamma=2):
    loss1 = -tf.reduce_sum((1 - y_pred) ** gamma * y_true * tf.math.log(y_pred), axis=-1)
    loss2 = -tf.reduce_sum(y_pred ** gamma * (1 - y_true) * tf.math.log(1 - y_pred), axis=-1)
    return alpha * loss1 + (1 - alpha) * loss2


def yolo_loss(targets, predictions, neg_pos_ratio=3):
    pos_mask = targets[:, :, 0]
    neg_mask = 1 - pos_mask
    reg_loss = smooth_L1lose(targets[:, :, 1:5], predictions[:, :, 1:5])
    reg_loss = reg_loss * pos_mask
    reg_loss = tf.reduce_mean(tf.reduce_sum(reg_loss, axis=-1))
    cls_loss = softmax_cross_entropy(targets[:, :, 5:], predictions[:, :, 5:])
    cls_loss = cls_loss * pos_mask
    cls_loss = tf.reduce_mean(tf.reduce_sum(cls_loss, axis=-1))
    # focal loss
    # obj_loss = focal_loss(targets[:, :, 0], predictions[:, :, 0])
    # obj_loss = tf.reduce_mean(obj_loss)
    #
    obj_loss = sigmoid_cross_entropy(targets[:, :, 0], predictions[:, :, 0])
    obj_loss_pos = obj_loss * pos_mask
    obj_loss_neg = obj_loss * neg_mask
    n_pos = tf.reduce_sum(pos_mask, axis=-1)
    n_neg = tf.reduce_sum(neg_mask, axis=-1)
    n_neg = tf.minimum(n_pos * neg_pos_ratio, n_neg)
    obj_loss_neg = tf.sort(obj_loss_neg, axis=-1, direction='DESCENDING')
    neg_mask = tf.range(obj_loss_neg.shape[1], dtype=tf.float32)[tf.newaxis, :] < n_neg[:, tf.newaxis]
    obj_loss_pos = tf.reduce_sum(obj_loss_pos, axis=-1)
    obj_loss_neg = tf.reduce_sum(obj_loss_neg * tf.cast(neg_mask, tf.float32), axis=-1)
    obj_loss = tf.reduce_mean(obj_loss_pos + obj_loss_neg)

    return obj_loss, reg_loss, cls_loss, obj_loss + reg_loss + cls_loss


class YOLOV3:
    def __init__(self, input_shape, n_classes=20):
        self.n_classes = n_classes
        self.inputs = keras.Input(shape=input_shape)
        feat1, feat2, feat3 = darknet53(self.inputs)
        self.outputs = prediction_conv(feat1, feat2, feat3, n_classes)
        self.model = keras.Model(inputs=self.inputs, outputs=self.outputs)
        self.trainable_availables = self.model.trainable_variables
        self.loss = yolo_loss

    def __call__(self, images):
        return self.model(images)

    def save(self, model_path):
        self.model.save(model_path)

    def load(self, model_path):
        self.model.load_weights(model_path)

    def predict_single_image(self, image):
        return self.predict(image[np.newaxis, :, :, :])[0]

    def predict(self, images):
        output = self.model(images)

    # print(YOLOV3([416,416,3]).model.summary())

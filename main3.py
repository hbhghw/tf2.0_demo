import os, cv2, sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt


def build_model(n_classes):  # ResNet50
    def conv_bn_relu(inputs, filters, kernel_size, strides, padding, bn=True, act='relu'):
        x = layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, padding=padding)(inputs)
        if bn:
            x = layers.BatchNormalization()(x)
        if act:
            x = layers.Activation(act)(x)

        return x

    def conv_block(inputs, filters, strides=2, padding='same'):
        shortcut = conv_bn_relu(inputs, filters * 4, 1, strides=strides, padding=padding, bn=False, act=None)

        x = conv_bn_relu(inputs, filters, 1, strides=strides, padding=padding)
        x = conv_bn_relu(x, filters, 3, strides=1, padding=padding)
        x = conv_bn_relu(x, filters * 4, 1, strides=1, padding=padding, act=None)
        x = layers.Add()([x, shortcut])
        return x

    def identity_block(inputs, filters, strides=1, padding='same'):
        shortcut = inputs
        x = conv_bn_relu(inputs, filters, 1, strides=strides, padding=padding)
        x = conv_bn_relu(x, filters, 3, strides=strides, padding=padding)
        x = conv_bn_relu(x, filters * 4, 1, strides=strides, padding=padding, act=None)
        x = layers.Add()([x, shortcut])
        return x

    def res_block(inputs, repeats, filters, strides=2):
        x = conv_block(inputs, filters, strides=strides)
        for i in range(repeats - 1):
            x = identity_block(x, filters)
        return x

    inputs = keras.Input(shape=[224, 224, 3], dtype='float32')
    x = conv_bn_relu(inputs, filters=64, kernel_size=7, strides=2, padding='same')

    x = layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)
    # [3,4,6,3]
    x = res_block(x, 3, 64, strides=1)
    x = res_block(x, 4, 128)
    x = res_block(x, 6, 256)
    x = res_block(x, 3, 512)

    x = layers.Activation('relu')(x)  # [batch_size,7,7,2048]
    x = layers.GlobalAveragePooling2D()(x)  # keras mode :[batch_size,2048]
    x = layers.Dense(units=512, activation='relu')(x)
    x = layers.Dense(units=128, activation='relu')(x)
    x = layers.Dense(units=32, activation='relu')(x)
    # x = layers.Dense(units=classes,activation='softmax')(x)  # [n,classes]
    x = layers.Dense(units=n_classes)(x)  # logits,[batch_size,n]

    model = Model(inputs=[inputs], outputs=[x])
    # print(model.summary())
    return model


def create_dataset(image_files, labels, flag='train'):
    def map_func(x, y):
        img = tf.io.read_file(x)
        img = tf.image.decode_jpeg(img, channels=3)  # tf.uint8 range[0,255]
        img = tf.image.convert_image_dtype(img, tf.float32)  # tf.float32,range[0,1]
        img = img * 2 - 1  # tf.float32 range[-1,1]
        img = tf.image.resize(img, (224, 224))
        # img = tf.image.random_brightness(img,0.5) #augment here
        return img, y

    dataset = tf.data.Dataset.from_tensor_slices((image_files, labels))
    dataset = dataset.map(map_func)
    if flag == 'train':
        dataset = dataset.repeat()
    dataset = dataset.shuffle(1024).batch(32)
    return dataset


def readData(dirs, train_val_rate=0.7):
    image_files = []  # class 0
    labels = []
    for class_i, dir_ in enumerate(dirs):
        for file in os.listdir(dir_):
            image_files.append(os.path.join(dir_, file))
            labels.append(class_i)

    assert len(image_files) == len(labels)

    image_files, labels = np.array(image_files), np.array(labels)
    indices = np.arange(len(labels))
    np.random.shuffle(indices)
    train_indices = indices[:int(len(indices) * train_val_rate)]
    val_indices = indices[int(len(indices) * train_val_rate):]
    train_images, train_labels = image_files[train_indices], labels[train_indices]
    val_images, val_labels = image_files[val_indices], labels[val_indices]

    return train_images, train_labels, val_images, val_labels


def compute_loss_with_logits(y_true, y_pred):
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))


def compute_acc(y_true, y_pred):
    return tf.reduce_mean(tf.cast(tf.equal(y_true, tf.argmax(y_pred, axis=-1)), tf.float32))


def train(dirs):
    n_classes = len(dirs)
    save_file = ''
    for d in dirs:
        save_file = save_file + '_' + d.split('/')[-1]
    save_file = save_file[1:] + '.h5'

    train_images, train_labels, val_images, val_labels = readData(dirs)
    train_dataset = create_dataset(train_images, train_labels)
    val_dataset = create_dataset(val_images, val_labels, flag='val')

    model = build_model(n_classes)
    opt = keras.optimizers.Adam(0.0001)
    # compute_loss = keras.metrics.BinaryCrossentropy()
    # compute_acc = keras.metrics.BinaryAccuracy()

    step = 0
    history = []
    for data, labels in train_dataset:
        step += 1
        with tf.GradientTape() as tape:
            y_pred = model(data)
            loss = compute_loss_with_logits(labels, y_pred)

        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))

        if step % 2 == 0:
            train_acc = compute_acc(labels, y_pred)
            total = 0
            correct = 0
            for batch_val_data, batch_val_labels in val_dataset:
                total += batch_val_labels.shape[0]
                pred = model.predict(batch_val_data)
                correct += np.sum(batch_val_labels.numpy() == np.argmax(pred, axis=-1))
            val_acc = correct / total
            print('step:', step, ' loss:', loss.numpy(), ' train acc:', train_acc.numpy(), ' validate acc:',
                  val_acc)
            # early stopping
            history.append(val_acc)
        stop = False
        if len(history) >= 5:
            stop = True
            for h in history[-5:]:
                if h < 0.95:
                    stop = False
                    break
        if stop or step == 2000:
            model.save('models/' + save_file)
            break

    plt.plot([i for i in range(len(history))], history)
    plt.show()


def evaluate(model_path, imgs_path):
    '''
    evaluate a trained model
    :param model_path: model file path
    :param imgs_path: directory of images
    :return: classify results
    '''
    model = keras.models.load_model(model_path)
    labels = model_path.split('/')[-1].split('_')
    results = []
    for img_file in os.listdir(imgs_path):
        img = cv2.imread(os.path.join(imgs_path, img_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224)) / 255.  # range [0,1]
        img = img * 2 - 1  # range [-1,1]
        pred = model.predict(np.expand_dims(img, axis=0))
        pred = np.argmax(pred, axis=-1)[0]
        results.append(labels[pred])  # 0 or 1
    return results


def predict_single_image(model_path, img_path):
    model = keras.models.load_model(model_path)
    labels = model_path[:-3].split('/')[-1].split('_')

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224)) / 255.  # range [0,1]
    img = img * 2 - 1  # range [-1,1]
    pred = model.predict(np.expand_dims(img, axis=0))
    pred_class = np.argmax(pred, axis=-1)[0]
    return labels[pred_class]


if __name__ == '__main__':
    # parser = ArgumentParser()
    # parser.add_argument('-a', '--dira', help='images directory for class a', required=True)
    # parser.add_argument('-b', '--dirb', help='images directory for class b', required=True)
    # parser.add_argument('-vs', '--validate_steps', default=20)
    # parser.add_argument('-bs', '--batch_size', default=32)
    # parser.add_argument('-s', '--steps', default=2000)
    # args = parser.parse_args()
    train(sys.argv[1:])
    # evaluate('models/model1.h5', 'DATA/select/problem')
    # print(predict_single_image('models/normal_problem.h5','DATA/select/problem/CK_16_3.jpeg'))

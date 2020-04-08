import tensorflow as tf
from tensorflow import keras
from dataset import PascalVOCDataset, get_data_list
from model import SSD300

model = SSD300()
train_imgs, train_anns, val_imgs, val_anns, test_imgs, tes_anns = get_data_list()
train_dataset = PascalVOCDataset(split='train',images=train_imgs,annotations=train_anns, batch_size=1)
opt = keras.optimizers.Adam(1e-4)
for i in range(2000):
    imgs, true_locs, true_cls = next(train_dataset)
    with tf.GradientTape() as tape:
        pred_locs, pred_cls = model(imgs)
        loss, _, _, _ = model.loss(true_locs, true_cls, pred_locs, pred_cls)
    gradients = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))
    if i%20==0:
        print(i, ' LOSS:', loss.numpy())
model.save('ssd300.h5')

from dataset import ImageDataset, get_data_list
from model import YOLOV3
import tensorflow as tf
from tensorflow import keras

train_images, train_annotations, val_images, val_annotations, test_images, test_annotations = get_data_list()

dataset = ImageDataset(train_images, train_annotations,batch_size=1,input_shape=(416,416))
model = YOLOV3(input_shape=(416,416,3))
opt = keras.optimizers.Adam(1e-4)

for i in range(2000):
    images,labels = next(dataset)
    with tf.GradientTape() as tape:
        predictions = model(images)
        obj_loss,reg_loss,cls_loss,loss = model.loss(labels,predictions)
    grads = tape.gradient(loss,model.trainable_availables)
    opt.apply_gradients(zip(grads,model.trainable_availables))
    if i%10==0:
        print(i,f'obj_loss={obj_loss},reg_loss={reg_loss},cls_loss={cls_loss},total loss={loss}')

model.save('v1.h5')
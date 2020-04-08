import numpy as np
import tensorflow as tf
from tensorflow import keras


def l1loss(labels, preds):
    return tf.reduce_sum(tf.abs(labels - preds), axis=-1)


def smooth_l1loss(labels, preds):
    v = tf.abs(labels - preds)
    ret = tf.where(v > 1, v - 0.5, 0.5 * tf.square(v))
    return tf.reduce_sum(ret, axis=-1)


def compute_crossentropy(labels, logits):
    logits = tf.clip_by_value(logits, 1e-8, 1 - 1e-8)
    return -tf.reduce_sum(labels * tf.math.log(logits) + (1 - labels) * tf.math.log(1 - logits), axis=-1)


def relu(x):
    return keras.layers.ReLU()(x)


def conv(x, filters, kernel_size=3, stride=1, padding='same'):
    return keras.layers.Conv2D(filters, kernel_size, stride, padding=padding)(x)


def maxPooling(x):
    return keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='same')(x)


def reshape(x, shape):
    return keras.layers.Reshape(shape)(x)


def VGGBase(inputs):
    x = relu(conv(inputs, 64))
    x = relu(conv(x, 64))
    x = maxPooling(x)
    x = relu(conv(x, 128))
    x = relu(conv(x, 128))
    x = maxPooling(x)
    x = relu(conv(x, 256))
    x = relu(conv(x, 256))
    x = relu(conv(x, 256))
    x = maxPooling(x)
    x = relu(conv(x, 512))
    x = relu(conv(x, 512))
    x = relu(conv(x, 512))
    feat_4 = x
    x = maxPooling(x)
    x = relu(conv(x, 512))
    x = relu(conv(x, 512))
    x = relu(conv(x, 512))
    x = keras.layers.MaxPooling2D(pool_size=3, strides=1, padding='same')(x)
    x = relu(keras.layers.Conv2D(1024, 3, 1, padding='same', dilation_rate=6)(x))
    feat_7 = relu(conv(x, 1024, kernel_size=1))
    return feat_4, feat_7


def AuxiliaryConvolutions(feat_7):
    x = relu(conv(feat_7, 256, kernel_size=1, padding='valid'))
    x = relu(conv(x, 512, kernel_size=3, stride=2))
    feat_8 = x
    x = relu(conv(x, 128, kernel_size=1, padding='valid'))
    x = relu(conv(x, 256, kernel_size=3, stride=2))
    feat_9 = x
    x = relu(conv(x, 128, kernel_size=1, padding='valid'))
    x = relu(conv(x, 256, kernel_size=3, padding='valid'))
    feat_10 = x
    x = relu(conv(x, 128, kernel_size=1, padding='valid'))
    feat_11 = relu(conv(x, 256, kernel_size=3, padding='valid'))
    return feat_8, feat_9, feat_10, feat_11


def predictionConvolutions(f4, f7, f8, f9, f10, f11):
    n_boxes = {'f4': 4, 'f7': 6, 'f8': 6, 'f9': 6, 'f10': 4, 'f11': 4}
    n_classes = 21
    loc_f4 = conv(f4, n_boxes['f4'] * 4, kernel_size=3)
    loc_f4 = reshape(loc_f4, [-1, 4])
    loc_f7 = conv(f7, n_boxes['f7'] * 4, kernel_size=3)
    loc_f7 = reshape(loc_f7, [-1, 4])
    loc_f8 = conv(f8, n_boxes['f8'] * 4, kernel_size=3)
    loc_f8 = reshape(loc_f8, [-1, 4])
    loc_f9 = conv(f9, n_boxes['f9'] * 4, kernel_size=3)
    loc_f9 = reshape(loc_f9, [-1, 4])
    loc_f10 = conv(f10, n_boxes['f10'] * 4, kernel_size=3)
    loc_f10 = reshape(loc_f10, [-1, 4])
    loc_f11 = conv(f11, n_boxes['f11'] * 4, kernel_size=3)
    loc_f11 = reshape(loc_f11, [-1, 4])
    cls_f4 = conv(f4, n_boxes['f4'] * n_classes)
    cls_f4 = reshape(cls_f4, [-1, n_classes])
    cls_f7 = conv(f7, n_boxes['f7'] * n_classes)
    cls_f7 = reshape(cls_f7, [-1, n_classes])
    cls_f8 = conv(f8, n_boxes['f8'] * n_classes)
    cls_f8 = reshape(cls_f8, [-1, n_classes])
    cls_f9 = conv(f9, n_boxes['f9'] * n_classes)
    cls_f9 = reshape(cls_f9, [-1, n_classes])
    cls_f10 = conv(f10, n_boxes['f10'] * n_classes)
    cls_f10 = reshape(cls_f10, [-1, n_classes])
    cls_f11 = conv(f11, n_boxes['f11'] * n_classes)
    cls_f11 = reshape(cls_f11, [-1, n_classes])
    locs = tf.concat([loc_f4, loc_f7, loc_f8, loc_f9, loc_f10, loc_f11], axis=1)
    cls = tf.concat([cls_f4, cls_f7, cls_f8, cls_f9, cls_f10, cls_f11], axis=1)
    return locs, cls


# inputs = keras.Input([300, 300, 3])
# f4, f7 = VGGBase(inputs)
# f8, f9, f10, f11 = AuxiliaryConvolutions(f7)
class SSD300:
    def __init__(self, input_shape=(300, 300, 3), n_classes=21):
        self.n_classes = n_classes
        self.neg_pos_ratio = 3
        self.rescale_factors = tf.Variable(20 * tf.ones([1, 1, 1, 512], tf.float32), trainable=True)
        self.inputs = keras.Input(input_shape)
        self.f4, self.f7 = VGGBase(self.inputs)
        norm = tf.sqrt(tf.reduce_sum(tf.square(self.f4), axis=-1, keepdims=True))
        self.f4 = self.f4 / norm
        self.f4 = self.f4 * self.rescale_factors
        self.f8, self.f9, self.f10, self.f11 = AuxiliaryConvolutions(self.f7)
        self.pred_locs, self.pred_cls = predictionConvolutions(self.f4, self.f7, self.f8, self.f9, self.f10, self.f11)
        self.pred_cls = tf.nn.softmax(self.pred_cls, axis=-1)
        self.model = keras.Model(inputs=self.inputs, outputs=[self.pred_locs, self.pred_cls])
        self.trainable_variables = self.model.trainable_variables
        # print(self.model.summary())

    def __call__(self, imgs):
        return self.model(imgs)

    def loss(self, true_locs, true_cls, pred_locs, pred_cls):
        reg_loss = smooth_l1loss(true_locs, pred_locs)
        neg_mask = true_cls[..., 0]  # background
        pos_mask = 1 - neg_mask
        reg_loss_for_pos = pos_mask * reg_loss
        cls_loss = compute_crossentropy(true_cls, pred_cls)
        cls_loss_for_pos = pos_mask * cls_loss
        cls_loss_for_neg = neg_mask * cls_loss
        # indices = tf.argsort(cls_loss_for_neg, direction='DESCENDING')
        # cls_loss_for_neg = cls_loss_for_neg[indices]
        cls_loss_for_neg = tf.sort(cls_loss_for_neg, axis=-1, direction='DESCENDING')
        n_pos = tf.reduce_sum(pos_mask, axis=-1)  # [batch,]
        n_neg = tf.reduce_sum(neg_mask, axis=-1)
        n_neg = tf.minimum(n_neg, n_pos * self.neg_pos_ratio)
        cls_mask = np.arange(cls_loss_for_neg.shape[1])[np.newaxis, :]  # [1,n_anchors]
        n_neg = tf.expand_dims(n_neg, axis=-1)  # [batch,1]
        cls_mask = cls_mask < n_neg
        cls_loss_for_neg = cls_loss_for_neg * tf.cast(cls_mask, tf.float32)
        reg_loss = tf.reduce_mean(tf.reduce_sum(reg_loss_for_pos, axis=1))
        cls_loss_for_pos = tf.reduce_mean(tf.reduce_sum(cls_loss_for_pos, 1))
        cls_loss_for_neg = tf.reduce_mean(tf.reduce_sum(cls_loss_for_neg, 1))
        return reg_loss + cls_loss_for_pos + cls_loss_for_neg, reg_loss, cls_loss_for_pos, cls_loss_for_neg

    def save(self, model_path):
        self.model.save(model_path)

    def load(self, model_path):
        self.model.load_weights(model_path)

    def detect_single_image(self, image, visualize=False):
        return self.detect_imgs(image[np.newaxis, ...], visualize)

    def detect_imgs(self, images, visualize=False):
        import cv2
        try:
            n_anchors = len(self.anchors_cxcy)
        except:
            from dataset import generate_all_anchors

            self.anchors_cxcy,self.anchors_xy = generate_all_anchors()

        pred_locs, pred_probs = self.model(images)
        batch_size = images.shape[0]
        for i in range(batch_size):
            pred_loc, pred_prob = pred_locs[i].numpy(), pred_probs[i].numpy()
            pred_loc[:,2:] = self.anchors_cxcy[:,2:] * np.exp(pred_loc[:,2:])
            pred_loc[:,:2] = self.anchors_cxcy[:,:2] + self.anchors_cxcy[:,2:] * pred_loc[:,:2]
            pred_loc = np.concatenate([pred_loc[:,:2]-pred_loc[:,2:]/2,pred_loc[:,:2]+pred_loc[:,2:]/2],axis=-1)
            pred_class = np.argmax(pred_prob, axis=1)
            pred_score = pred_prob[np.arange(len(pred_prob)), pred_class]
            pos_mask = pred_class > 0
            pred_class: np.ndarray
            pred_loc, pred_class, pred_score = pred_loc[pos_mask], pred_class[pos_mask], pred_score[pos_mask]
            indices = nms(pred_score,pred_loc)
            pred_loc = pred_loc[indices]*300
            pred_loc = pred_loc.astype(int)
            pred_class = pred_class[indices]
            pred_score = pred_score[indices]
            if visualize:
                img = images[i]
                for j,box in enumerate(pred_loc):
                    cv2.rectangle(img,(box[0],box[1]),(box[2],box[3]),(0,255,0),1)
                cv2.imshow('a',img)
                cv2.waitKey(5000)

def nms(scores,boxes,iou_threshold=0.5): #non maximum suppress
    keep = []
    indices = np.argsort(scores)[::-1]
    areas = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])

    while indices:
        i = indices[0]
        keep.append(i)
        # indices = indices[1:]

        xmin = np.maximum(boxes[i][0],boxes[indices[1:],0])
        ymin = np.maximum(boxes[i][1],boxes[indices[1:],1])
        xmax = np.minimum(boxes[i][2],boxes[indices[1:],2])
        ymax = np.minimum(boxes[i][3],boxes[indices[1:],3])
        w,h = np.maximum(0,xmax-xmin),np.maximum(0,ymax-ymin)
        intersection = w * h
        iou = intersection / (areas[i]+areas[indices[1:]]-intersection)
        mask = iou<iou_threshold
        indices = indices[1:][mask]
    return np.array(keep,dtype=int)

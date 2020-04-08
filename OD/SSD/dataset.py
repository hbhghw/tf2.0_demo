import tensorflow as tf
import numpy as np
import cv2
from xml.etree import ElementTree as ET

voc_labels = (
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
    'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
label_map = {k: v + 1 for v, k in enumerate(voc_labels)}
label_map['background'] = 0
rev_label_map = {v: k for k, v in label_map.items()}


def parse_annotation(file):
    root = ET.parse(file).getroot()
    labels = []
    boxes = []
    difficulties = []
    for obj in root.iter('object'):
        difficult = int(obj.find('difficult').text == '1')
        label = obj.find('name').text.lower().strip()
        if label not in label_map:
            continue
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text) - 1
        ymin = int(bbox.find('ymin').text) - 1
        xmax = int(bbox.find('xmax').text) - 1
        ymax = int(bbox.find('ymax').text) - 1

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label_map[label])
        difficulties.append(difficult)
    return {'boxes': boxes, 'labels': labels, 'difficulties': difficulties}


def compute_iou(box, anchors):
    # anchors:[n_anchors,4],[xmin,ymin,xmax,ymax]
    x1, y1, x2, y2 = box
    xmin = np.maximum(x1, anchors[:, 0])
    ymin = np.maximum(y1, anchors[:, 1])
    xmax = np.minimum(x2, anchors[:, 2])
    ymax = np.minimum(y2, anchors[:, 3])
    w, h = np.maximum(xmax - xmin, 0), np.maximum(ymax - ymin, 0)
    intersection = w * h
    area_box = (x2 - x1) * (y2 - y1)
    area_anchors = (anchors[:, 2] - anchors[:, 0]) * (anchors[:, 3] - anchors[:, 1])
    union = area_box + area_anchors - intersection
    return intersection / union


def compute_regression_target(box, anchors_cxcy):
    x1, y1, x2, y2 = box
    cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
    reg = np.zeros_like(anchors_cxcy, dtype=np.float)  # regression target
    reg[:, :2] = (np.array([cx, cy]) - anchors_cxcy[:, :2]) / anchors_cxcy[:, 2:]
    reg[:, 2:] = np.log(np.array([w, h]) / anchors_cxcy[:, 2:])
    return reg


def generate_true_labels(boxes, classes, anchors_cxcy, anchors_xy, n_classes=21, iou_threshold=0.6):
    # boxes: [x1,y1,x2,y2]
    n_anchors = anchors_cxcy.shape[0]
    true_loc = np.zeros([n_anchors, 4])
    true_cls = np.zeros([n_anchors, n_classes])
    true_cls[:, 0] = 1
    for box, c in zip(boxes, classes):
        iou = compute_iou(box, anchors_xy)
        indices = iou > iou_threshold
        reg = compute_regression_target(box, anchors_cxcy)
        true_cls[indices, 0] = 0
        true_cls[indices, c] = 1
        true_loc[indices, :] = reg[indices, :]
    return true_loc, true_cls


def generate_all_anchors():
    sqrt2 = np.sqrt(2)
    sqrt3 = np.sqrt(3)
    feature_maps = [[38, 38],  # [fh,fw]
                    [19, 19],
                    [10, 10],
                    [5, 5],
                    [3, 3],
                    [1, 1]]
    wh = [
        [[0.1, 0.1], [np.sqrt(0.1 * 0.2), np.sqrt(0.1 * 0.2)], [0.1 / sqrt2, 0.1 * sqrt2], [0.1 * sqrt2, 0.1 / sqrt2]],
        [[0.2, 0.2], [np.sqrt(0.2 * 0.375), np.sqrt(0.2 * 0.375)], [0.2 / sqrt2, 0.2 * sqrt2],
         [0.2 * sqrt2, 0.2 / sqrt2], [0.2 / sqrt3, 0.2 * sqrt3], [0.2 * sqrt3, 0.3 / sqrt3]],
        [[0.375, 0.375], [np.sqrt(0.375 * 0.55), np.sqrt(0.375 * 0.55)], [0.375 / sqrt2, 0.375 * sqrt2],
         [0.375 * sqrt2, 0.375 / sqrt2], [0.375 / sqrt3, 0.375 * sqrt3], [0.375 * sqrt3, 0.375 / sqrt3]],
        [[0.55, 0.55], [np.sqrt(0.55 * 0.725), np.sqrt(0.55 * 0.725)], [0.55 / sqrt2, 0.55 * sqrt2],
         [0.55 * sqrt2, 0.55 / sqrt2], [0.55 / sqrt3, 0.55 * sqrt3], [0.55 * sqrt3, 0.55 / sqrt3]],
        [[0.725, 0.725], [np.sqrt(0.725 * 0.9), np.sqrt(0.725 * 0.9)], [0.725 / sqrt2, 0.725 * sqrt2],
         [0.725 * sqrt2, 0.725 / sqrt2]],
        [[0.9, 0.9], [1.0, 1.0], [0.9 / sqrt2, 0.9 * sqrt2], [0.9 * sqrt2, 0.9 / sqrt2]]]

    all_anchors = []
    for fmap, anchor_wh in zip(feature_maps, wh):
        fh, fw = fmap
        anchors = np.zeros([fh, fw, len(anchor_wh), 4])
        x, y = np.arange(fw), np.arange(fh)
        x, y = np.meshgrid(x, y)
        x, y = x[:, :, np.newaxis, np.newaxis], y[:, :, np.newaxis, np.newaxis]
        xy = np.concatenate([x, y], axis=-1)
        xy = (xy + 0.5) / [fw, fh]
        anchors[:, :, :, :2] += xy
        anchors[:, :, :, 2:] += anchor_wh
        anchors = np.reshape(anchors, [-1, 4])
        all_anchors.append(anchors)
    all_anchors_cxcy = np.concatenate(all_anchors, axis=0)
    all_anchors_xy = np.concatenate([all_anchors_cxcy[:, :2] - all_anchors_cxcy[:, 2:] / 2,
                                     all_anchors_cxcy[:, :2] + all_anchors_cxcy[:, 2:] / 2], axis=-1)
    return all_anchors_cxcy, all_anchors_xy  # [cx,cy,w,h],[x1,y1,x2,y2]


class PascalVOCDataset:
    def __init__(self, split='train', images=None, annotations=None, keep_difficult=False, batch_size=1, shuffle=False,
                 data_argu=False):
        self.split = split
        self.images = images
        self.annotations = annotations
        self.keep_difficult = keep_difficult
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = True
        self.data_argu = data_argu

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.anchors_cxcy, self.anchors_xy = generate_all_anchors()
        self.build_dataset()

    def build_dataset(self):
        self.total_len = len(self.images)
        self.cur = 0
        self.indices = np.arange(self.total_len)
        np.random.shuffle(self.indices)

        def gen():
            while True:
                if self.cur >= self.total_len:
                    np.random.shuffle(self.indices)
                    self.cur = 0
                image = cv2.imread(self.images[self.cur])
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype('float32')
                objects = parse_annotation(self.annotations[self.cur])
                boxes = np.array(objects['boxes']).astype('float32')
                labels = np.array(objects['labels'])
                difficulties = np.array(objects['difficulties'])
                if not self.keep_difficult:
                    mask = difficulties < 1
                    boxes = boxes[mask]
                    labels = labels[mask]
                    difficulties = difficulties[mask]
                if self.split == 'TRAIN' and self.data_argu:
                    # todo: image data argument
                    pass
                height, width, _ = image.shape
                image = cv2.resize(image, (300, 300))
                image /= 255.
                image = (image - self.mean) / self.std
                boxes[:, [0, 2]] /= width
                boxes[:, [1, 3]] /= height
                boxes, labels = generate_true_labels(boxes, labels, self.anchors_cxcy, self.anchors_xy)
                self.cur += 1
                yield image, boxes, labels

        self.dataset = tf.data.Dataset.from_generator(gen, output_types=(tf.float32, tf.float32, tf.float32))
        self.dataset = self.dataset.batch(self.batch_size)
        self.dataset = iter(self.dataset)

    def __next__(self):
        return next(self.dataset)


def get_data_list():
    train_images, train_annotations = [], []
    val_images, val_annotations = [], []
    test_images, test_annotations = [], []
    with open('data/VOC2007_trainval/ImageSets/Main/train.txt') as f:
        for idx in f.readlines():
            train_images.append(f'data/VOC2007_trainval/JPEGImages/{idx.strip()}.jpg')
            train_annotations.append(f'data/VOC2007_trainval/Annotations/{idx.strip()}.xml')
    with open('data/VOC2007_trainval/ImageSets/Main/val.txt') as f:
        for idx in f.readlines():
            val_images.append(f'data/VOC2007_trainval/JPEGImages/{idx.strip()}.jpg')
            val_annotations.append(f'data/VOC2007_trainval/Annotations/{idx.strip()}.xml')
    with open('data/VOC2012_trainval/ImageSets/Main/train.txt') as f:
        for idx in f.readlines():
            train_images.append(f'data/VOC2012_trainval/JPEGImages/{idx.strip()}.jpg')
            train_annotations.append(f'data/VOC2012_trainval/Annotations/{idx.strip()}.xml')
    with open('data/VOC2012_trainval/ImageSets/Main/val.txt') as f:
        for idx in f.readlines():
            val_images.append(f'data/VOC2012_trainval/JPEGImages/{idx.strip()}.jpg')
            val_annotations.append(f'data/VOC2012_trainval/Annotations/{idx.strip()}.xml')
    with open('data/VOC2007_test/ImageSets/Main/test.txt') as f:
        for idx in f.readlines():
            test_images.append(f'data/VOC2007_test/JPEGImages/{idx.strip()}.jpg')
            test_annotations.append(f'data/VOC2007_test/Annotations/{idx.strip()}.xml')
    return train_images, train_annotations, val_images, val_annotations, test_images, test_annotations

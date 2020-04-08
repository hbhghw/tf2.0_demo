import numpy as np
import cv2
import xml.etree.ElementTree as ET
import tensorflow as tf

voc_labels = (
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
    'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
label_map = {k: v for v, k in enumerate(voc_labels)}
rev_label_map = {v: k for k, v in label_map.items()}


def parse_annotation(xml_file):
    root = ET.parse(xml_file).getroot()
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
    return np.array(boxes, dtype=np.float), np.array(labels), np.array(difficulties)


def compute_iou(box, anchors_xy):
    xmin = np.maximum(box[0], anchors_xy[:, 0])
    ymin = np.maximum(box[1], anchors_xy[:, 1])
    xmax = np.minimum(box[2], anchors_xy[:, 2])
    ymax = np.minimum(box[3], anchors_xy[:, 3])
    w, h = np.maximum(0, xmax - xmin), np.maximum(0, ymax - ymin)
    intersection = w * h
    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_anhors = (anchors_xy[:, 2] - anchors_xy[:, 0]) * (anchors_xy[:, 3] - anchors_xy[:, 1])
    union = area_box + area_anhors - intersection
    return intersection / union


def compute_regression_target(box, anchors_cxcy):
    x1, y1, x2, y2 = box
    cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
    reg = np.zeros_like(anchors_cxcy, dtype=np.float)
    reg[:, :2] = (np.array([cx, cy]) - anchors_cxcy[:, :2]) / anchors_cxcy[:, 2:]
    reg[:, 2:] = np.log(np.array([w, h]) / anchors_cxcy[:, 2:])
    return reg


def generate_true_labels(boxes, labels, anchors_cxcy, anchors_xy, n_classes=len(label_map), iou_threshold=0.3):
    n_anchors = anchors_cxcy.shape[0]
    ret_labels = np.zeros([n_anchors, 1 + 4 + n_classes])
    for box, c in zip(boxes, labels):
        iou = compute_iou(box, anchors_xy)
        mask = iou > iou_threshold
        ret_labels[mask, 0] = 1
        ret_labels[mask, 5 + c] = 1
        regression = compute_regression_target(box, anchors_cxcy)
        ret_labels[mask, 1:5] = regression[mask, :]
    return ret_labels


def generate_all_anchors(input_shape=(416, 416)):
    yolo3_feature_maps = [[52, 52], [26, 26], [13, 13]]  # [fh,fw]
    yolo3_wh = [[[10, 13], [16, 30], [33, 23]],
                [[30, 61], [62, 45], [59, 119]],
                [[116, 90], [156, 198], [373, 326]]]
    yolo3_wh = np.array(yolo3_wh) / input_shape

    all_anchors = []
    for fmap, anchor_wh in zip(yolo3_feature_maps, yolo3_wh):
        fh, fw = fmap
        anchors = np.zeros([fh, fw, len(anchor_wh), 4])
        x, y = np.arange(fw), np.arange(fh)
        x, y = np.meshgrid(x, y)
        xy = np.concatenate([x[:, :, np.newaxis, np.newaxis], y[:, :, np.newaxis, np.newaxis]], axis=-1)
        xy = (xy + 0.5) / [fw, fh]
        anchors[:, :, :, :2] += xy
        anchors[:, :, :, 2:] += anchor_wh
        anchors = np.reshape(anchors, [-1, 4])
        all_anchors.append(anchors)
    all_anchors_cxcy = np.concatenate(all_anchors, axis=0)
    all_anchors_xy = np.concatenate([all_anchors_cxcy[:, :2] - all_anchors_cxcy[:, 2:] / 2,
                                     all_anchors_cxcy[:, :2] + all_anchors_cxcy[:, 2:] / 2], axis=-1)
    return all_anchors_cxcy, all_anchors_xy


class ImageDataset:
    def __init__(self, img_paths, xml_paths, batch_size=1, data_argument=False, keep_difficulties=False,
                 input_shape=(300, 300)):
        assert len(img_paths) == len(xml_paths)
        self.img_paths = img_paths
        self.xml_paths = xml_paths
        self.batch_size = batch_size
        self.data_argu = data_argument
        self.keep_difficulties = keep_difficulties
        self.input_shape = input_shape

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.anchors_cxcy, self.anchors_xy = generate_all_anchors()
        self.build_dataset()

    def build_dataset(self):
        self.total_size = len(self.img_paths)
        self.cur = 0
        self.indices = np.arange(self.total_size)
        np.random.shuffle(self.indices)

        def generator():
            while True:
                if self.cur >= self.total_size:
                    self.cur = 0
                    np.random.shuffle(self.indices)
                index = self.indices[self.cur]
                img = cv2.imread(self.img_paths[index])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype('float')
                boxes, labels, difficulties = parse_annotation(self.xml_paths[index])
                if not self.keep_difficulties:
                    keep = difficulties < 0.5
                    boxes, labels, difficulties = boxes[keep], labels[keep], difficulties[keep]
                if self.data_argu:
                    pass
                h, w, _ = img.shape
                img = cv2.resize(img, self.input_shape)
                img = img / 255.
                img: np.ndarray
                img = (img - self.mean) / self.std
                boxes[:, [0, 2]] /= w
                boxes[:, [1, 3]] /= h
                labels = generate_true_labels(boxes, labels, self.anchors_cxcy, self.anchors_xy)
                self.cur += 1
                yield img, labels  # [batch,h,w,3],[batch,n_anchors,25]

        self.dataset = tf.data.Dataset.from_generator(generator, output_types=(tf.float32, tf.float32))
        self.dataset = self.dataset.batch(self.batch_size)
        self.dataset = iter(self.dataset)

    def __next__(self):
        return next(self.dataset)


def get_data_list():
    train_images, train_annotations = [], []
    val_images, val_annotations = [], []
    test_images, test_annotations = [], []
    with open('E:\python\ObjectDetection\SSD\data\VOC2007_trainval/ImageSets/Main/train.txt') as f:
        for idx in f.readlines():
            train_images.append(f'E:\python\ObjectDetection\SSD\data\VOC2007_trainval/JPEGImages/{idx.strip()}.jpg')
            train_annotations.append(
                f'E:\python\ObjectDetection\SSD\data\VOC2007_trainval/Annotations/{idx.strip()}.xml')
    with open('E:\python\ObjectDetection\SSD\data\VOC2007_trainval/ImageSets/Main/val.txt') as f:
        for idx in f.readlines():
            val_images.append(f'E:\python\ObjectDetection\SSD\data\VOC2007_trainval/JPEGImages/{idx.strip()}.jpg')
            val_annotations.append(f'E:\python\ObjectDetection\SSD\data\VOC2007_trainval/Annotations/{idx.strip()}.xml')
    with open('E:\python\ObjectDetection\SSD\data\VOC2012_trainval/ImageSets/Main/train.txt') as f:
        for idx in f.readlines():
            train_images.append(f'E:\python\ObjectDetection\SSD\data\VOC2012_trainval/JPEGImages/{idx.strip()}.jpg')
            train_annotations.append(
                f'E:\python\ObjectDetection\SSD\data\VOC2012_trainval/Annotations/{idx.strip()}.xml')
    with open('E:\python\ObjectDetection\SSD\data\VOC2012_trainval/ImageSets/Main/val.txt') as f:
        for idx in f.readlines():
            val_images.append(f'E:\python\ObjectDetection\SSD\data\VOC2012_trainval/JPEGImages/{idx.strip()}.jpg')
            val_annotations.append(f'E:\python\ObjectDetection\SSD\data\VOC2012_trainval/Annotations/{idx.strip()}.xml')
    with open('E:\python\ObjectDetection\SSD\data\VOC2007_test/ImageSets/Main/test.txt') as f:
        for idx in f.readlines():
            test_images.append(f'E:\python\ObjectDetection\SSD\data\VOC2007_test/JPEGImages/{idx.strip()}.jpg')
            test_annotations.append(f'E:\python\ObjectDetection\SSD\data\VOC2007_test/Annotations/{idx.strip()}.xml')
    return train_images, train_annotations, val_images, val_annotations, test_images, test_annotations


if __name__ == '__main__':
    np.random.seed(666)
    train_images, train_annotations, val_images, val_annotations, test_images, test_annotations = get_data_list()
    dataset = ImageDataset(train_images, train_annotations)
    imgs, boxes, labels = next(dataset)
    img = imgs[0].numpy()
    img = img * dataset.std + dataset.mean
    h, w, _ = img.shape
    boxes = boxes.numpy()[0]
    boxes[:, :2] = dataset.anchors_cxcy[:, :2] + dataset.anchors_cxcy[:, 2:] * boxes[:, :2]
    boxes[:, 2:] = dataset.anchors_cxcy[:, 2:] * np.exp(boxes[:, 2:])
    boxes = np.concatenate([boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2], axis=-1)
    labels = labels[0]
    classes = np.argmax(labels, axis=-1)
    pos = classes > 0
    boxes = boxes[pos]
    for box in boxes:
        cv2.rectangle(img, (int(box[0] * w), int(box[1] * h)), (int(box[2] * w), int(box[3] * h)), (0, 255, 255), 2)
    cv2.imshow('a', img)
    cv2.waitKey(6000)

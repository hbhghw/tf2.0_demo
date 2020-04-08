from model import SSD300
import cv2
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

model = SSD300()
model.load('v1.h5')
img_path = r'data/VOC2007_test/JPEGImages/000004.jpg'
img = cv2.imread(img_path)
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB).astype('float')
img = cv2.resize(img, (300, 300))
img /= 255.
img = (img - mean) / std
model.detect_single_image(img,visualize=True)
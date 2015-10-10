import cv2
import numpy as np
from matplotlib import pyplot as plt

names = ['0.png', '1.jpg', '2.jpg', '3.png', '4.jpg', '5.png']
names = ['images/maze' + name for name in names]

img = cv2.imread(names[0])
plt.subplot(1,1,0)
plt.imshow(img)

img = cv2.resize(img, (400,500));
plt.subplot(1,1,0)
plt.imshow(img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret,gray = cv2.threshold(gray,127,255,0)
gray2 = gray.copy()
mask = np.zeros(gray.shape,np.uint8)
print(mask)


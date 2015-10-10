
import cv2
import numpy as np
from matplotlib import pyplot as plt

names = ['0.png', '1.jpg', '2.jpg', '3.png', '4.jpg', '5.png']
names = ['images/maze' + name for name in names]

for name in names:
	img = cv2.imread(name)
	plt.subplot(1,1,0)
	plt.imshow(img)
	plt.show()
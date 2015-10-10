
import cv2
import numpy as np
from matplotlib import pyplot as plt
import copy
import time


names = ['0', '1.jpg', '2.jpg', '3.png', '4.jpg', '5.png']
names = ['noisy_images/noise{}.jpg'.format(i) for i in range(6)]
'''
names = ['0.png', '1.jpg', '2.jpg', '3.png', '4.jpg', '5.png']
names = ['images/maze' + name for name in names]
'''
def filter_(img):
	# Otsu's thresholding
	ret,th = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	kernel = np.ones((3,3), np.float32)/9
	th3 = cv2.filter2D(img, -1, kernel)
	return th



def zhangsuen(img):
	#pseudocode: http://rosettacode.org/wiki/Zhang-Suen_thinning_algorithm
	# black = 1, white = 0
	shape = img.shape
	step = 1;

	# optimization step
	check_pixels = set()
	for row in range(1, shape[0] - 1):
		for col in range(1, shape[1] - 1):
			check_pixels.add((row, col))

	while len(check_pixels) > 0:
		selected_pixels = set()
		if step == 1:
			new_pixels = set()

#		order = list(check_pixels)
#		order.sort(key=lambda x: x[1])
#		order.sort(key=lambda x: x[0])
		for pixel in check_pixels:
			row, col = pixel

			if row < 1 or row > shape[0] - 2:
				continue
			if col < 1 or col > shape[1] - 2:
				continue

			# if condition 0 not met
			if img[row, col] == 0:
				continue

			p2 = img[row-1, col]
			p3 = img[row-1, col+1]
			p4 = img[row, col+1]
			p5 = img[row+1, col+1]
			p6 = img[row+1, col]
			p7 = img[row+1, col-1]
			p8 = img[row, col-1]
			p9 = img[row-1, col-1]

			#import pdb; pdb.set_trace()
			blackPix = map(lambda x: x != 0, [p2, p3, p4, p5, p6, p7, p8, p9])
			blackNum = sum([1 if b else 0 for b in blackPix])
			if blackNum < 2 or blackNum> 6:
				continue

			trans = [p2,p3,p4,p5,p6,p7,p8,p9]
			trans2 = [p3,p4,p5,p6,p7,p8,p9,p2]
			trans = map(lambda x: x[0] ==0 and x[1] != 0, zip(trans, trans2))
			numTrans = sum([1 if t else 0 for t in trans])
			
			# if condition 2 not met
			if numTrans != 1:
				continue

			if (step == 1 and (p4== 0 or p6==0 or (p2==0 and p8==0)))\
				or (step == 0 and (p2 == 0 or p8 == 0 or (p4==0 and p6==0))):
				selected_pixels.add((row, col))
				

		for pixel in selected_pixels:
			row, col = pixel
			img[row, col] = 0
			new_pixels.add((row-1, col))
			new_pixels.add((row-1, col+1))
			new_pixels.add((row, col+1))
			new_pixels.add((row+1, col+1))
			new_pixels.add((row+1, col))
			new_pixels.add((row+1, col-1))
			new_pixels.add((row, col-1))
			new_pixels.add((row-1, col-1))

		step = (step + 1) % 2

		if step == 1:
			check_pixels = new_pixels

	return img

# uses vector things to try to find the beginning and end in the maze
# assumes that there is a gap to the outside when running this
def find_ends(img):
	THRESHOLD = 10
	height, width = img.shape

	# search in 5 pixel chunks, top and bottom
	depths = []
	depths2 = []
	for i in range(width / 5):
		total = 0
		total2 = 0
		for j in range(5):
			for k in range(height):
				if img[k, i*5+j]:
					total += k
					break
			for k in reversed(range(height)):
				if img[k, i*5+j]:
					total2 += k
					break
		depths.append(total)
		depths2.append(total2)

	return depths, depths2

img = cv2.imread('hao.jpg', cv2.IMREAD_GRAYSCALE)

# gets the area of interest
filtered = filter_(img)
# fix I found on stack overflow:


z = zhangsuen(filtered)
#import pdb; pdb.set_trace()


plt.subplot(2, 2, 3)
plt.title('cropped_img')
plt.imshow(z, cmap='gray')

plt.show()

for name in names:
	img = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
	img_copy = img.copy()

	# gets the area of interest
	filtered = filter_(img)
	# fix I found on stack overflow:
	# http://stackoverflow.com/questions/20743850/python-opencv-error-finding-contours
	ret,thresh = cv2.threshold(filtered.copy(),127,255,cv2.THRESH_BINARY_INV)
	contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	cont_per = map(lambda x: (x, cv2.arcLength(x, True)), contours)
	cont_per.sort(key=lambda x: -x[1])
	longest_2 = zip(*cont_per[0:2])[0]
	rects = map(lambda x: cv2.boundingRect(x), longest_2)

	# finds rectangle bounding maze
	DELTA = 5
	min_x0, min_y0, max_x0, max_y0 =\
		rects[0][0], rects[0][1], rects[0][0] + rects[0][2], rects[0][1] + rects[0][3]
	min_x1, min_y1, max_x1, max_y1 =\
		rects[1][0], rects[1][1], rects[1][0] + rects[1][2], rects[1][1] + rects[1][3]
	min_x = min(min_x0, min_x1) - DELTA
	min_y = min(min_y0, min_y1) - DELTA
	max_x = max(max_x0, max_x1) + DELTA
	max_y = max(max_y0, max_y1) + DELTA

	cropped_img = filtered[min_y: max_y, min_x: max_x]

	# now process the cropped_img
	t0 = time.time()
	z = zhangsuen(cropped_img)
	print time.time() - t0
	#import pdb; pdb.set_trace()

	# draws bounding boxes around rectangles
	filtered_copy = filtered.copy()
	cv2.rectangle(filtered_copy,(min_x,min_y),(max_x,max_y),0,1)

	#filtered_copy = filtered.copy()

	plt.subplot(2, 2, 0)
	plt.title('original')
	plt.imshow(img, cmap='gray')
	plt.subplot(2, 2, 1)
	plt.title('with rectangle')
	plt.imshow(filtered_copy, cmap='gray')
	plt.subplot(2, 2, 2)
	plt.title('filtered')
	plt.imshow(filtered, cmap='gray')
	plt.subplot(2, 2, 3)
	plt.title('cropped_img')
	plt.imshow(z, cmap='gray')

	plt.show()


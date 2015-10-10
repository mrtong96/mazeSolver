
import cv2
import numpy as np
from matplotlib import pyplot as plt
import copy
<<<<<<< HEAD
import time
=======
from scipy import weave
import sys
>>>>>>> f942dbb7c7cdefe882b0a29edc0096107340f355


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
	th = cv2.filter2D(th, -1, kernel)
	return th
'''
# does the A thing
def get_image_set(img):
	zeros = []
	width = img[0].size
	height = img.size / width
	for y in range(height):
		for x in range(width):
			if img[y][x] == 0:
				zeros.append((x,y))

	A = np.zeros((height, width), dtype=np.uint8)

	ERR = .000001 # for robustness against FLOP errors
	for i, z0 in enumerate(zeros):
		print i, len(zeros)
		for z1 in zeros[i:]:
			x0, y0 = z0
			x1, y1 = z1

			if x1 - x0:
				m = 1.0 * (y1 - y0) / (x1 - x0)
				# checks y intersections
				for i in range(1, x1 - x0):
					x_coords = [x0 + i - 1, x0 + i]
					y_res = 1.0 * i * m + y0
					y_coords = [int(y_res)]
					if abs(y_res - int(y_res)) < ERR:
						if y_res > int(y_res):
							y_coords.append(int(y_res) + 1)
						else:
							y_coords.append(int(y_res) - 1)
					for x in x_coords:
						for y in y_coords:
							A[y][x] |= 0xFF

			if y1 - y0:
				m_t = 1.0 * (x1 - x0) / (y1 - y0)
				# checks x intersections
				for i in range(1, y1 - y0):
					y_coords = [y0 + i - 1, y0 + i]
					x_res = 1.0 * i * m_t + x0
					x_coords = [int(x_res)]
					if abs(x_res - int(x_res)) < ERR:
						if x_res > int(x_res):
							x_coords.append(int(x_res) + 1)
						else:
							x_coords.append(int(x_res) - 1)
					for x in x_coords:
						for y in y_coords:
							A[y][x] |= 0xFF
	return A
'''

def zhangsuen(img):
	#pseudocode: http://rosettacode.org/wiki/Zhang-Suen_thinning_algorithm
	# black = 1, white = 0
	shape = img.shape
	step = 1;
	changed = True
	
	# optimization step
	check_pixels = set()
	for row in range(1, shape[0] - 1):
		for col in range(1, shape[1] - 1):
			check_pixels.add((row, col))

	while len(check_pixels) > 0:
		if step == 1:
			new_pixels = set()

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

<<<<<<< HEAD
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


for name in names[:1]:
=======
def zhangsuen(img):
	#pseudocode: http://rosettacode.org/wiki/Zhang-Suen_thinning_algorithm
	shape = img.shape
	step = 1;
	changed = True
	while(changed):
		changed = False
		for row in range(1, shape[0] - 1):
			for col in range(1, shape[1] - 1):
				isBlack = (img[row,col] == 0)
				numTransitions = (img[row - 1, col - 1] == 255 or img[row - 1, col] == 255) + \
					(img[row - 1, col] == 255 or img[row - 1, col + 1] == 255) + \
					(img[row - 1, col + 1] == 255 or img[row, col + 1] == 255) + \
					(img[row, col + 1] == 255 or img[row + 1, col + 1] == 255) + \
					(img[row + 1, col + 1] == 255 or img[row + 1, col] == 255) + \
					(img[row + 1, col] == 255 or img[row + 1, col - 1] == 255) + \
					(img[row + 1, col - 1] == 255 or img[row, col - 1] == 255) + \
					(img[row, col - 1] == 255 or img[row - 1, col - 1] == 255)
				numBlackNeighbors = (img[row - 1, col - 1] == 0) + (img[row - 1, col] == 0) + (img[row - 1, col + 1] == 0) + (img[row, col - 1] == 0) + (img[row, col + 1] == 0) + (img[row + 1, col - 1] == 0) + (img[row + 1, col] == 0) + (img[row + 1, col + 1] == 0)
				if(step == 1):
					fuckThis = (img[row - 1, col] == 255 or img[row, col + 1] == 255 or img[row + 1, col] == 255)
					fuckThis2 = (img[row, col - 1] == 255 or img[row, col + 1] == 255 or img[row + 1, col] == 255)
					step = 2
				elif(step == 2):
					fuckThis = (img[row - 1, col] == 255 or img[row, col + 1] == 255 or img[row, col - 1] == 255)
					fuckThis2 = (img[row, col - 1] == 255 or img[row - 1, col] == 255 or img[row + 1, col] == 255)
					step = 1
				if(isBlack and numBlackNeighbors >= 2 and numBlackNeighbors <= 6 and numTransitions == 1 and fuckThis and fuckThis2):
					img[row,col] == 255
					changed = True
	return img

for name in names:
>>>>>>> f942dbb7c7cdefe882b0a29edc0096107340f355
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
	z = zhangsuen(cropped_img.copy())
	#img_set = get_image_set(cropped_img)
	print time.time() - t0
	#import pdb; pdb.set_trace()

	# draws bounding boxes around rectangles
	filtered_copy = filtered.copy()
	cv2.rectangle(filtered_copy,(min_x,min_y),(max_x,max_y),0,1)

	#filtered_copy = filtered.copy()

	zhang_suen = zhangsuen(cropped_img.copy())


	plt.subplot(2, 3, 0)
	plt.title('original')
	plt.imshow(img, cmap='gray')
<<<<<<< HEAD
	plt.subplot(2, 2, 1)
	plt.title('with rectangle')
	plt.imshow(filtered_copy, cmap='gray')
	plt.subplot(2, 2, 2)
	plt.title('filtered')
=======
	plt.subplot(2, 3, 1)
	plt.title('filtered')
	plt.imshow(filtered_copy, cmap='gray')
	plt.subplot(2, 3, 2)
	plt.title('with rectangle')
>>>>>>> f942dbb7c7cdefe882b0a29edc0096107340f355
	plt.imshow(filtered, cmap='gray')
	plt.subplot(2, 3, 3)
	plt.title('cropped_img')
<<<<<<< HEAD
	plt.imshow(z, cmap='gray')

=======
	plt.imshow(cropped_img, cmap='gray')
	plt.subplot(2, 3, 4)
	plt.title('zhang suen')
	plt.imshow(zhang_suen, cmap='gray')
>>>>>>> f942dbb7c7cdefe882b0a29edc0096107340f355
	plt.show()
	print(cropped_img)


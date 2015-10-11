
import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
import Queue

# does all of the graphing stuff

class ImageSolver():
	def __init__(self, img):
		# crop the image
		self.height, self.width = img.shape
		self.img = np.zeros((self.height, self.width), dtype=np.uint8)

		pairs = [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]
		for row in range(self.height):
			for col in range(self.width):
				if img[row, col] == 0:
					continue
				pixel_values = []
				for dx, dy in pairs:
					if row + dy < 0 or row + dy >= self.height:
						pixel_values.append(0)
					elif col + dx < 0 or col + dx >= self.width:
						pixel_values.append(0)
					else:
						pixel_values.append(img[row + dy, col + dx])
				black_to_white = map(lambda x: pixel_values[x] != 0 and pixel_values[x+1] == 0, range(8))
				black_to_white = sum([1 if b else 0 for b in black_to_white])
				self.img[row, col] = black_to_white

		# points of interest
		self.POI = []
		for row in range(self.height):
			for col in range(self.width):
				# if is junction or dead end
				if self.img[row, col] > 2 or self.img[row, col] == 1:
					self.POI.append((row, col))

	# pixels from POI0 -> POI1 
	def find_route(self, POI0, POI1):
		img_copy = self.img.copy()
		for i in range(self.height):
			self.img[i, 0] = 0
			self.img[i, self.width - 1] = 0
		for i in range(self.width):
			self.img[0, i] = 0
			self.img[self.height - 1, i] = 0
		self.img[POI0[0], POI0[1]] = 1		
		self.img[POI1[0], POI1[1]] = 1		
		#import pdb; pdb.set_trace()
		if POI0 not in self.POI or POI1 not in self.POI:
			return

		closed_set = set()
		queue = Queue.Queue()
		closed_set.add(POI0)
		successors = self.get_successors(POI0)
		for suc in successors:
			queue.put((suc, [POI0]))
			#closed_set.add(suc)

		# in the form of location, previous locations
		while not queue.empty():
			loc, prev_locs = queue.get()
			if loc == POI1:
				self.img = img_copy
				return prev_locs
			if loc not in closed_set:
				closed_set.add(loc)
				successors = self.get_successors(loc)
				for suc in successors:
					new_locs = prev_locs + [loc]
					queue.put((suc, new_locs))
		self.img = img_copy
		return None



	# returns list of next pts given this point
	def get_successors(self, pt):
		#import pdb; pdb.set_trace()
		pts = []
		pairs = [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]
		for dy, dx in pairs:
			new_pt = (pt[0] + dy, pt[1] + dx)
			if new_pt[0] < 0 or new_pt[0] >= self.height or\
					new_pt[1] < 0 or new_pt[1] >= self.width:
				continue
			elif self.img[new_pt[0], new_pt[1]]:
				pts.append(new_pt)
		return pts




	def highlightPOIs(self,img):
		clone = img.copy()
		for point in self.POI:
			cv2.circle(clone, (point[1], point[0]), 6, (255,0,0), 3)
		return clone
	'''
	def highlightPaths(self,img):
		clone = img.copy()
		for key in self.paths.keys():
			for pixel in self.paths[key]:
				clone[pixel] = [0,255,0]
		return clone
	'''
	def identifyEnds(self,img):
		self.ends = []
		for poi in self.POI:
			if poi[1] == self.width - 1 or poi[1] == 0 or poi[0] == 0 or poi[0] == self.height - 1:
				self.ends.append(poi);
				self.img[poi] = 1
		if len(self.ends) != 2:
			print('Lenny for a watermelon you run into a lot more trouble than you should')
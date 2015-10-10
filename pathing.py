
import cv2
import numpy as np
from matplotlib import pyplot as plt
import time

# does all of the graphing stuff

class ImageSolver():
	def __init__(self, img):
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

		# dict, for POI u, v, stores path pixels from u <-> v
		self.paths = {}
		# iterate through every POI and store paths
		for i, poi in enumerate(self.POI):
			row, col = poi
			# figure out what points are part of what paths
			pairs = [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)]
			pixel_groups = []

			for i, (dx, dy) in enumerate(pairs):
				# not on table, assume black
				if row + dy < 0 or row + dy >= self.height or\
						col + dx < 0 or col + dx >= self.width:
					continue
				# part of path in neighbors
				elif self.img[row + dy, col + dx]:
					foundGroup = False
					for group in pixel_groups:
						next_pair = pairs[(i+1)%8]
						prev_pair = pairs[(i-1)%8]
						next_pair = row + next_pair[1], col + next_pair[0]
						prev_pair = row + prev_pair[1], col + prev_pair[0]

						if next_pair in group:
							group.add((row+dy, col+dx))
							foundGroup = True
							break
						elif prev_pair in group:
							group.add((row+dy, col+dx))
							foundGroup = True
							break
					if not foundGroup:
						to_append = set()
						to_append.add((row+dy, col+dx))
						pixel_groups.append(to_append)

			# now go forth along the path, essentially dfs
			for group in pixel_groups:
				closed_set = group.union(set([poi]))
				to_expand = group.copy()

				# do some expanding
				new_junction = None
				while len(to_expand) > 0:
					pix = to_expand.pop()
					pairs = [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)]
					for dx, dy in pairs:
						new_pix = (pix[0] + dy, pix[1] + dx)
						# if invalid location, continue
						if new_pix[0] < 0 or new_pix[0] >= self.height or\
								new_pix[1] < 0 or new_pix[1] >= self.width:
							continue
						# else not found before
						elif new_pix not in closed_set:
							closed_set.add(new_pix)
							# found next junction
							if self.img[new_pix[0], new_pix[1]] >= 3:
								new_junction = new_pix
							if new_junction == None:
								to_expand.add(new_pix)

				self.paths[(poi, new_junction)] = closed_set


		print len(self.POI)



t0 = time.time()
name = 'noisy_images/noise0_out.png'
img = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
plt.subplot(1,2,0)
plt.title('orig')
plt.imshow(img)
solver = ImageSolver(img)
plt.subplot(1,2,1)
plt.title('new')
plt.imshow(solver.img)
cv2.imwrite('tmp.jpg', solver.img)
print time.time() - t0
#plt.show()
#import pdb; pdb.set_trace()


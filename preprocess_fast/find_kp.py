from discoverlib import geom, grid_index
import util

import cv2
import math
import numpy
import os.path
import random
import skimage.io

matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_L1)

class ActiveKP(object):
	def __init__(self, thetas, drone_point, frame_info, desc):
		self.thetas = [thetas]
		self.drone_points = [drone_point]
		self.frame_infos = [frame_info]
		self.frame_range = (frame_info[0], frame_info[0])
		self.desc = desc
		self.n = 1
		self.age = 0
		self.ttl = 0

	def update_frame_range(self, frame_idx):
		lower, upper = self.frame_range
		if frame_idx < lower:
			self.frame_range = (frame_idx, upper)
			return True
		elif frame_idx > upper:
			self.frame_range = (lower, frame_idx)
			return True
		else:
			return False

	def add(self, thetas, drone_point, frame_info, desc):
		if not self.update_frame_range(frame_info[0]):
			return
		self.thetas.append(thetas)
		self.drone_points.append(drone_point)
		self.frame_infos.append(frame_info)
		self.desc = (self.n*self.desc + desc)/(self.n+1)
		self.n += 1
		self.ttl = 0

	def tick(self):
		self.age += 1
		self.ttl += 1

	def valid(self):
		if self.age >= 1 and self.n < 2:
			return False
		if (self.age >= 10 or self.ttl >= 4) and self.n < 5:
			return False
		return True

	def old(self):
		return self.ttl >= 4

	def compute_pos(self):
		# need to solve a set of linear equations like this:
		# x,y,h: kp position
		# a, b: drone position
		# theta_x, theta_y: observation offset
		# * x + 0y - h*tan(theta_x) - a = 0
		# * 0x + y - h*tan(theta_y) - b = 0
		num_equations = 2*self.n
		A = numpy.zeros((num_equations, 3), dtype='float32')
		b = numpy.zeros((num_equations,), dtype='float32')
		for i in range(self.n):
			thetas = self.thetas[i]
			drone_point = self.drone_points[i]
			A[2*i, :] = [1, 0, -math.tan(thetas[0])]
			b[2*i] = drone_point[0]
			A[2*i+1, :] = [0, 1, -math.tan(thetas[1])]
			b[2*i+1] = drone_point[1]
		x, _, _, _ = numpy.linalg.lstsq(A, b)
		return int(x[0]), int(x[1]), int(x[2])

class FinalKP(object):
	def __init__(self, p, h, desc):
		self.p = p
		self.h = h
		self.desc = desc
		self.n = 1

	def add(self, p, h, desc):
		self.p = (
			(self.n*self.p[0] + p[0])/(self.n+1),
			(self.n*self.p[1] + p[1])/(self.n+1),
		)
		self.h = (self.n*self.h + h)/(self.n+1)
		self.desc = (self.n*self.desc + desc)/(self.n+1)
		self.n += 1

counter = 0

def incorporate_good_kps(cfg, good_kps, final_kps, index):
	global counter
	# incorporate old active_kps into final_kps
	# (1) get a rectangle around good_kps
	# (2) find all final_kps in the rectangle
	# (3) use knnmatch to match them
	# (4) update the ones that match, add the ones that don't
	if len(good_kps) == 0:
		return

	poslist = []
	good_bounding_rect = None
	for idx, kp in enumerate(good_kps):
		x, y, h = kp.compute_pos()
		poslist.append((x, y, h))

		if kp.frame_infos[0][0] > 100 and idx == 0 and False:
			for i, frame_info in enumerate(kp.frame_infos):
				frame = skimage.io.imread(os.path.join(cfg.video_path, util.pad6(frame_info[0]) + '.jpg'))
				frame = cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))
				frame[frame_info[2]-2:frame_info[2]+2, frame_info[1]-2:frame_info[1]+2, :] = [255, 0, 0]
				print('...', frame_info)
				skimage.io.imsave('/home/ubuntu/vis/{}-{}.jpg'.format(counter, i), frame)

			base_im = skimage.io.imread(os.path.join(cfg.data_path, 'ortho.jpg'))
			base_im[y-10:y+10, x-10:x+10, :] = [255, 0, 0]
			skimage.io.imsave('/home/ubuntu/vis/{}-x.jpg'.format(counter), base_im)
			counter += 1

		p = geom.Point(x, y)
		if good_bounding_rect is None:
			good_bounding_rect = p.bounds()
		else:
			good_bounding_rect = good_bounding_rect.extend(p)

	final_cmp = [final_kps[i] for i in index.search(good_bounding_rect)]
	final_cmp += random.sample(final_kps, min(len(final_kps), 5000))

	matches1 = {}
	good_descs = numpy.array([kp.desc for kp in good_kps], dtype='float32')
	if len(final_cmp) >= 2:
		final_descs = numpy.array([kp.desc for kp in final_cmp], dtype='float32')
		for match in matcher.knnMatch(queryDescriptors=good_descs, trainDescriptors=final_descs, k=2):
			m, n = match
			if m.distance > 0.6*n.distance:
				continue

			if m.queryIdx not in matches1:
				matches1[m.queryIdx] = []
			matches1[m.queryIdx].append(m)

	matches = {}
	for l in matches1.values():
		l.sort(key=lambda m: m.distance)
		if len(l) > 1 and l[0].distance > 0.6*l[1].distance:
			continue
		m = l[0]
		x, y, h = poslist[m.queryIdx]
		if util.eucl((x, y), final_cmp[m.trainIdx].p) > 100:
			continue
		if abs(h-final_cmp[m.trainIdx].h) > 100:
			continue
		matches[m.queryIdx] = m

	for i, kp in enumerate(good_kps):
		if i in matches:
			m = matches[i]
			final_cmp[m.trainIdx].add(poslist[i][0:2], poslist[i][2], good_descs[i])
		else:
			point = geom.Point(poslist[i][0], poslist[i][1])
			index.insert(point, len(final_kps))
			final_kps.append(FinalKP(poslist[i][0:2], poslist[i][2], good_descs[i]))

def get_thetas(cfg, im, bounds, query_points):
	# estimate the drone height in world coordinates from its bbox
	# tan(fov/2) = width/2 / h
	drone_center = (int(sum([p[0] for p in bounds])/4), int(sum([p[1] for p in bounds])/4))
	b_width = abs(numpy.mean([util.eucl(bounds[0], bounds[1]), util.eucl(bounds[2], bounds[3])]))
	b_height = abs(numpy.mean([util.eucl(bounds[1], bounds[2]), util.eucl(bounds[0], bounds[3])]))
	width_estimate = numpy.mean([b_width, b_height*cfg.aspect_ratio])
	drone_height = width_estimate/2.0/math.tan(cfg.horizontal_field_of_view/2)

	src_points = numpy.array([
		[0, 0],
		[im.shape[1], 0],
		[im.shape[1], im.shape[0]],
		[0, im.shape[0]],
	], dtype='float32').reshape(-1, 1, 2)
	dst_points = numpy.array(bounds, dtype='float32').reshape(-1, 1, 2)
	H, _ = cv2.findHomography(src_points, dst_points)

	# now get thetas along world axes from drone_center
	# tan(theta_x) = dx / h
	keypoints = numpy.array(query_points, dtype='float32').reshape(-1, 1, 2)
	keypoints = cv2.perspectiveTransform(keypoints, H).reshape(-1, 2)
	keypoints = [(int(pt[0]), int(pt[1])) for pt in keypoints]
	keypoint_angles = [(
		math.atan((pt[0]-drone_center[0])/drone_height),
		math.atan((pt[1]-drone_center[1])/drone_height),
	) for pt in keypoints]
	return drone_center, keypoint_angles

def find_keypoints(cfg, frame_ims, sift_features, frame_bounds):
	active_kps = []
	final_kps = []
	index = grid_index.GridIndex(128)

	def get_active_descs():
		if len(active_kps) == 0:
			return numpy.zeros((0, 128), dtype='float32')
		else:
			return numpy.array([kp.desc for kp in active_kps], dtype='float32')

	for frame_idx, (keypoints, descs) in enumerate(sift_features):
		if keypoints is None:
			continue

		bounds = frame_bounds[frame_idx]
		if bounds is None:
			continue

		print('find_keypoints: process {}'.format(frame_idx))
		keypoints = [(kp.pt[0], kp.pt[1]) for kp in keypoints]
		drone_center, thetas = get_thetas(cfg, frame_ims[frame_idx], bounds, keypoints)

		matches1 = {}
		for match in matcher.knnMatch(queryDescriptors=descs, trainDescriptors=get_active_descs(), k=2):
			if len(match) < 2:
				continue
			m, n = match
			if m.distance > 0.6*n.distance:
				continue

			if m.queryIdx not in matches1:
				matches1[m.queryIdx] = []
			matches1[m.queryIdx].append(m)

		matches = {}
		for l in matches1.values():
			l.sort(key=lambda m: m.distance)
			if len(l) > 1 and l[0].distance > 0.6*l[1].distance:
				continue
			m = l[0]
			# TODO: only match if the optical flow is close
			#if util.eucl(keypoints[m.queryIdx], active_kps[m.trainIdx].frame_infos[-1][1:3]) > 200:
			#	continue
			matches[m.queryIdx] = m

		# integrate into active_kps
		for kp in active_kps:
			kp.tick()

		# TODO: if multiple keypoints in current frame match, we should vote and take the one with shortest distance
		for i, desc in enumerate(descs):
			frame_info = (frame_idx, int(keypoints[i][0]), int(keypoints[i][1]))
			if i in matches:
				m = matches[i]
				active_kps[m.trainIdx].add(thetas[i], drone_center, frame_info, desc)
			else:
				active_kps.append(ActiveKP(thetas[i], drone_center, frame_info, desc))

		active_kps = [kp for kp in active_kps if kp.valid()]

		good_kps = [kp for kp in active_kps if kp.old()]
		active_kps = [kp for kp in active_kps if not kp.old()]
		incorporate_good_kps(cfg, good_kps, final_kps, index)

	return final_kps

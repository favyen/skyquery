from discoverlib import geom, grid_index

import sift

import cv2
import json
import math
import multiprocessing
import numpy
import os
from PIL import Image
import skimage.io
import sys

LK_PARAMETERS = dict(winSize=(21, 21), maxLevel=2, criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, 0.01))

def homography_from_flow(prev_homography, prev_gray, cur_gray):
	positions = []
	for i in range(0, prev_gray.shape[0]-50, 50):
		for j in range(0, prev_gray.shape[1]-50, 50):
			positions.append((i, j))
	positions_np = numpy.array(positions, dtype='float32').reshape(-1, 1, 2)

	def flip_pos(positions):
		return numpy.stack([positions[:, :, 1], positions[:, :, 0]], axis=2)

	next_positions, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, cur_gray, flip_pos(positions_np), None, **LK_PARAMETERS)
	if next_positions is None:
		return None

	next_positions = flip_pos(next_positions)
	differences = next_positions[:, 0, :] - positions_np[:, 0, :]
	differences_okay = differences[numpy.where(st[:, 0] == 1)]
	median = [numpy.median(differences_okay[:, 0]), numpy.median(differences_okay[:, 1])]
	good = (numpy.square(differences[:, 0] - median[0]) + numpy.square(differences[:, 1] - median[1])) < 16

	if float(numpy.count_nonzero(good)) / differences.shape[0] < 0.7:
		return None

	# translate previous homography based on the flow result
	translation = [numpy.median(differences[:, 0]), numpy.median(differences[:, 1])]
	H_translation = numpy.array([[1, 0, -translation[1]], [0, 1, -translation[0]], [0,0,1]], dtype='float32')
	return prev_homography.dot(H_translation)

def align(frames, sift_features, frame_bounds, final_kps):
	matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_L1)
	out_bounds = [None for _ in frame_bounds]

	base_desc = numpy.array([kp.desc for kp in final_kps], dtype='float32')
	index = grid_index.GridIndex(128)
	for i, kp in enumerate(final_kps):
		index.insert(geom.Point(kp.p[0], kp.p[1]), i)

	prev_gray = None
	prev_homography = None

	for frame_idx, im in enumerate(frames):
		if im is None:
			continue

		print('align_homography: process {}'.format(frame_idx))
		cur_bounds = frame_bounds[frame_idx]
		cur_homography = None
		cur_gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

		if prev_homography is not None and (sift_features[frame_idx][0] is None or frame_idx % 6 != 0):
			# don't have pre-computed SIFT features, so attempt to match using flow
			cur_homography = homography_from_flow(prev_homography, prev_gray, cur_gray)
		#	if cur_homography is not None:
				# flow is good -- then smoothing should work well too
				# so skip this frame
		#		prev_gray = cur_gray
		#		prev_homography = cur_homography
		#		continue
		elif prev_homography is None and sift_features[frame_idx][0] is None:
			# don't waste sift features on bad parts
			prev_gray = None
			prev_homography = None
			continue

		if cur_homography is None or True:
			keypoints, desc = sift_features[frame_idx]
			if keypoints is None:
				keypoints, desc = sift.compute_one(im)

			if cur_bounds is None:
				train_keypoints, train_desc = final_kps, base_desc
				#rect = None
				#prev_gray = None
				#prev_homography = None
				#continue
			else:
				#rect = util.bounds_to_rect(cur_bounds)
				#rect = rect.add_tol(2*const.MAX_SPEED)
				#indices = list(index.search(rect.add_tol(const.MAX_SPEED)))
				#train_keypoints = [final_kps[idx] for idx in indices]
				#train_desc = base_desc[indices]
				train_keypoints, train_desc = final_kps, base_desc
				#print(rect, len(train_keypoints))

			'''matches = matcher.knnMatch(queryDescriptors=desc, trainDescriptors=train_desc, k=2)
			#good = {}
			good = []
			for m, n in matches:
				kp = final_kps[m.trainIdx]
				if m.distance > 0.6*n.distance:
					continue
				elif not kp.okay:
					continue
				#elif rect is not None and not rect.contains(geom.Point(kp.p[0], kp.p[1])):
				#	continue
				#elif m.trainIdx in good and good[m.trainIdx].distance < m.distance:
				#	continue
				#good[m.trainIdx] = m
				good.append(m)

			#good = good.values()
			#if len(good) < 10:
			#	prev_gray = None
			#	prev_homography = None
			#	continue

			src_pts = numpy.float32([keypoints[m.queryIdx].pt for m in good]).reshape(-1,1,2)
			dst_pts = numpy.float32([final_kps[m.trainIdx].p for m in good]).reshape(-1,1,2)'''

			matches = matcher.knnMatch(queryDescriptors=train_desc, trainDescriptors=desc, k=2)
			good1 = {}
			for m, n in matches:
				kp = final_kps[m.queryIdx]
				if m.distance > 0.6*n.distance:
					continue
				elif not kp.okay:
					continue
				#elif rect is not None and not rect.contains(geom.Point(kp.p[0], kp.p[1])):
				#	continue
				#elif m.queryIdx in good and good[m.queryIdx].distance < m.distance:
				#	continue
				#good[m.queryIdx] = m
				if m.queryIdx not in good1:
					good1[m.queryIdx] = []
				good1[m.queryIdx].append(m)

			good = []
			for l in good1.values():
				l.sort(key=lambda m: m.distance)
				if len(l) > 1 and l[0].distance > 0.6*l[1].distance:
					continue
				good.append(l[0])

			if len(good) < 10:
				prev_gray = None
				prev_homography = None
				continue

			src_pts = numpy.float32([keypoints[m.trainIdx].pt for m in good]).reshape(-1,1,2)
			dst_pts = numpy.float32([final_kps[m.queryIdx].p for m in good]).reshape(-1,1,2)

			try:
				cur_homography, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
			except Exception as e:
				print('... warning: findHomography error: {}'.format(e))
				prev_gray = None
				prev_homography = None
				continue

		if cur_homography is None:
			print('... warning: findHomography failed')
			prev_gray = None
			prev_homography = None
			continue

		prev_gray = cur_gray
		prev_homography = cur_homography

		bound_points = numpy.array([
			[0, 0],
			[im.shape[1], 0],
			[im.shape[1], im.shape[0]],
			[0, im.shape[0]],
		], dtype='float32').reshape(-1, 1, 2)
		transformed_points = cv2.perspectiveTransform(bound_points, cur_homography)
		transformed_points = transformed_points[:, 0, :].astype('int').tolist()
		print('... got bounds: ', transformed_points)
		out_bounds[frame_idx] = transformed_points

	return out_bounds

import cv2
import json
import math
import multiprocessing
import numpy
import os, os.path
import skimage.io
import sys

video_path = sys.argv[1]
data_path = sys.argv[2]

sift = cv2.xfeatures2d.SIFT_create()
matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_L1)
base_im = skimage.io.imread(os.path.join(data_path, 'ortho.jpg'))
base_keypoints, base_desc = sift.detectAndCompute(base_im, None)

frame_bounds = []

def add_bounds(frame_idx, bounds):
	while frame_idx >= len(frame_bounds):
		frame_bounds.append(None)
	frame_bounds[frame_idx] = bounds

for fname in sorted(os.listdir(video_path)):
	if '.jpg' not in fname:
		continue
	frame_idx = int(fname.split('.jpg')[0])

	print('process {}'.format(frame_idx))
	frame = skimage.io.imread(os.path.join(video_path, fname))
	frame = cv2.resize(frame, (frame.shape[1]/2, frame.shape[0]/2))
	frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

	keypoints, desc = sift.detectAndCompute(frame, None)
	matches = matcher.knnMatch(queryDescriptors=base_desc, trainDescriptors=desc, k=2)
	good = []
	for m, n in matches:
		if m.distance < 0.6*n.distance:
			good.append(m)

	src_pts = numpy.float32([keypoints[m.trainIdx].pt for m in good]).reshape(-1,1,2)
	dst_pts = numpy.float32([base_keypoints[m.queryIdx].pt for m in good]).reshape(-1,1,2)

	try:
		H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
	except Exception as e:
		print('... warning: findHomography error: {}'.format(e))
		add_bounds(frame_idx, None)
		continue

	if H is None:
		print('... warning: findHomography failed')
		add_bounds(frame_idx, None)
		continue

	bound_points = numpy.array([
		[0, 0],
		[frame.shape[1], 0],
		[frame.shape[1], frame.shape[0]],
		[0, frame.shape[0]],
	], dtype='float32').reshape(-1, 1, 2)
	transformed_points = cv2.perspectiveTransform(bound_points, H)
	transformed_points = transformed_points[:, 0, :].astype('int').tolist()
	print('... got bounds: ', transformed_points)
	add_bounds(frame_idx, transformed_points)

with open(os.path.join(data_path, 'align-out.json'), 'w') as f:
	json.dump(frame_bounds, f)

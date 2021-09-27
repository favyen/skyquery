# this is for the pedestrian/cyclist detector
# for first stage we create pairs of frames (f1, f2)
# f1 is a frame and f2 is following frame modified to align with f1

import cv2
import json
import math
import multiprocessing
import numpy
import os, os.path
import skimage.io
import sys

video_path = sys.argv[1]
out_path = sys.argv[2]

fnames = [fname for fname in os.listdir(video_path) if '.jpg' in fname]
fnames = [fname for fname in fnames if int(fname.split('.jpg')[0]) % 5 == 0]
fnames.sort(key=lambda fname: int(fname.split('.jpg')[0]))

sift = cv2.xfeatures2d.SIFT_create()
matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_L1)

prev_fname, prev_frame, prev_keypoints, prev_desc = None, None, None, None
for fname in fnames:
	print(fname)
	frame = skimage.io.imread(os.path.join(video_path, fname))
	keypoints, desc = sift.detectAndCompute(frame, None)
	if prev_frame is not None:
		matches = matcher.knnMatch(queryDescriptors=prev_desc, trainDescriptors=desc, k=2)
		good = []
		for m, n in matches:
			if m.distance < 0.6*n.distance:
				good.append(m)

		src_pts = numpy.float32([keypoints[m.trainIdx].pt for m in good]).reshape(-1,1,2)
		dst_pts = numpy.float32([prev_keypoints[m.queryIdx].pt for m in good]).reshape(-1,1,2)

		try:
			H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
		except Exception as e:
			print('warning: exception on file {}: {}'.format(fname, e))
			prev_frame, prev_keypoints, prev_desc = None, None, None
			continue

		if H is None:
			print('warning: failed on file {}'.format(fname))
			prev_frame, prev_keypoints, prev_desc = None, None, None
			continue

		warped = cv2.warpPerspective(frame, H, (prev_frame.shape[1], prev_frame.shape[0]))
		skimage.io.imsave(os.path.join(out_path, prev_fname.replace('.jpg', '.prev.jpg')), prev_frame)
		skimage.io.imsave(os.path.join(out_path, prev_fname.replace('.jpg', '.next.jpg')), warped)

	prev_fname, prev_frame, prev_keypoints, prev_desc = fname, frame, keypoints, desc

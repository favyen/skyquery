import sys
sys.path.append('./python')
from discoverlib import geom

import cv2
import json
import math
import multiprocessing
import numpy
import os, os.path
import skimage.io
import skimage.measure

PADDING = 32

def get_frame_fname(video_path, frame_id):
	for fname in os.listdir(video_path):
		if '.jpg' not in fname:
			continue
		cur_id = int(fname.split('.jpg')[0])
		if frame_id == cur_id:
			return os.path.join(video_path, fname)
	return None

def points_from_poly_str(s):
	parts = s.split(' ')
	points = []
	for part in parts:
		x, y = part.split(',')
		points.append(geom.Point(float(x), float(y)))
	return points

def points_to_rect(points):
	rect = points[0].bounds()
	for p in points:
		rect = rect.extend(p)
	return rect

video_path = sys.argv[1]
frame1 = int(sys.argv[2])
frame2 = int(sys.argv[3])
poly1 = points_from_poly_str(sys.argv[4])
poly2 = points_from_poly_str(sys.argv[5])

im1 = skimage.io.imread(get_frame_fname(video_path, frame1))
im2 = skimage.io.imread(get_frame_fname(video_path, frame2))
bounds1 = geom.Rectangle(geom.Point(0, 0), geom.Point(im1.shape[1], im1.shape[0]))
bounds2 = geom.Rectangle(geom.Point(0, 0), geom.Point(im2.shape[1], im2.shape[0]))
rect1 = points_to_rect(poly1)
rect2 = points_to_rect(poly2)
big1 = bounds1.clip_rect(rect1.add_tol(PADDING))
big2 = bounds2.clip_rect(rect2.add_tol(PADDING))
crop1 = im1[big1.start.y:big1.end.y, big1.start.x:big1.end.x, :]
crop2 = im2[big2.start.y:big2.end.y, big2.start.x:big2.end.x, :]

sift = cv2.xfeatures2d.SIFT_create()
matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_L1)
kp1, desc1 = sift.detectAndCompute(crop1, None)
kp2, desc2 = sift.detectAndCompute(crop2, None)

matches = matcher.knnMatch(queryDescriptors=desc1, trainDescriptors=desc2, k=2)
good = []
for m, n in matches:
	if m.distance < 0.6*n.distance or True:
		good.append(m)

src_pts = numpy.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
dst_pts = numpy.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)

try:
	H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
except Exception as e:
	print('bad1')
	sys.exit(0)

if H is None:
	print('bad2')
	sys.exit(0)

warp2 = cv2.warpPerspective(crop2, H, (crop1.shape[1], crop1.shape[0]))

start = rect1.start.sub(big1.start)
end = start.add(rect1.lengths())
sm1 = crop1[start.y:end.y, start.x:end.x, :]
sm2 = warp2[start.y:end.y, start.x:end.x, :]
#Image.fromarray(sm1).save('out1.png')
#Image.fromarray(sm2).save('out2.png')

print(skimage.measure.compare_ssim(sm1, sm2, multichannel=True))

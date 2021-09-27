import sys
sys.path.append('./python')

from discoverlib import geom, grid_index
import align_homography
import find_kp
import sift
import util

import cv2
import json
import math
import numpy
import os, os.path
import random
import time
import skimage.io

class Config(object):
	# max_speed is in ortho-imagery resolution units which in example was originally 2cm/pixel but resized to 4cm/pixel
	def __init__(self, video_path, data_path, horizontal_field_of_view=63.0*math.pi/180, aspect_ratio=1920.0/1080.0, max_speed=75):
		self.video_path = video_path
		self.data_path = data_path
		self.horizontal_field_of_view = horizontal_field_of_view
		self.aspect_ratio = aspect_ratio
		self.max_speed = max_speed

def filter_bounds(frame_bounds):
	for frame_idx, bounds in enumerate(frame_bounds):
		if bounds is None:
			continue
		b_width = abs(numpy.mean([util.eucl(bounds[0], bounds[1]), util.eucl(bounds[2], bounds[3])]))
		b_height = abs(numpy.mean([util.eucl(bounds[1], bounds[2]), util.eucl(bounds[0], bounds[3])]))
		if abs(util.eucl(bounds[0], bounds[1]) - 1240) > 200:
			frame_bounds[frame_idx] = None
		if abs(util.eucl(bounds[2], bounds[3]) - 1240) > 200:
			frame_bounds[frame_idx] = None
		if abs(util.eucl(bounds[1], bounds[2]) - 700) > 200:
			frame_bounds[frame_idx] = None
		if abs(util.eucl(bounds[0], bounds[3]) - 700) > 200:
			frame_bounds[frame_idx] = None

def get_good_final_kps(final_kps):
	idx = grid_index.GridIndex(256)
	for kp in final_kps:
		idx.insert(geom.Point(kp.p[0], kp.p[1]), kp)
	good_kps = []
	for kps in idx.grid.values():
		for threshold in [5, 4, 3, 2]:
			kp_filter = [kp for kp in kps if kp.n >= threshold]
			if threshold > 2 and len(kp_filter) < 64:
				continue
			good_kps.extend(kp_filter)
			break
	return good_kps

def get_good2(final_kps_all):
	final_kps = []
	for kp in final_kps_all:
		kp.okay = kp.n >= 2 and kp.h > 1000
		if kp.okay or random.random() < 0.2:
			final_kps.append(kp)
	return final_kps

def get_good3(final_kps_all):
	for kp in final_kps_all:
		kp.okay = False
	#good_kps = [kp for kp in get_good_final_kps(final_kps_all) if kp.n >= 2 and kp.h > 1000]
	good_kps = [kp for kp in final_kps_all if kp.n >= 2 and kp.h > 1000]
	final_kps = []
	for kp in good_kps:
		kp.okay = True
		final_kps.append(kp)
	for kp in random.sample(final_kps_all, 5000):
		if kp.okay:
			continue
		kp.okay = False
		final_kps.append(kp)
	return final_kps

def smooth_bounds(frame_bounds):
	prev = [None for _ in frame_bounds]
	next = [None for _ in frame_bounds]
	for i in range(len(frame_bounds)):
		if frame_bounds[i] is not None:
			prev[i] = i
		elif i > 0:
			prev[i] = prev[i-1]
	for i in range(len(frame_bounds)-1, -1, -1):
		if frame_bounds[i] is not None:
			next[i] = i
		elif i+1 < len(frame_bounds):
			next[i] = next[i+1]
	for i, bounds in enumerate(frame_bounds):
		if bounds is not None or prev[i] is None or next[i] is None:
			continue
		p_rect = util.bounds_to_rect(frame_bounds[prev[i]])
		n_rect = util.bounds_to_rect(frame_bounds[next[i]])
		if p_rect.iou(n_rect) < 0.7:
			continue
		p = numpy.array(frame_bounds[prev[i]], dtype='float32')
		n = numpy.array(frame_bounds[next[i]], dtype='float32')
		pw = next[i]-i
		nw = i-prev[i]
		avg = (p*pw+n*nw)/(pw+nw)
		frame_bounds[i] = avg.astype('int32').tolist()

def score_bounds(frame_bounds):
	with open('/home/ubuntu/skyquery2/bounds/gps-5.json', 'r') as f:
		orig_bounds = json.load(f)
	filter_bounds(orig_bounds)
	num_missing = 0
	errors = []
	for i, bounds in enumerate(orig_bounds):
		if bounds is None:
			continue
		out = frame_bounds[i]
		if out is None:
			num_missing += 1
			continue
		errors.append(util.eucl(bounds[0], out[0]))
	print('missing={}, err={}'.format(num_missing, numpy.mean(errors)))

def smooth2(frame_bounds):
	def get_window(i):
		n = 2
		for n in range(2, 30):
			start = i-n
			end = i+n
			if start < 0:
				start = 0
			if end >= len(frame_bounds):
				end = len(frame_bounds)-1
			lowl = [(start+j, bounds) for (j, bounds) in enumerate(frame_bounds[start:i]) if bounds is not None]
			highl = [(i+j, bounds) for (j, bounds) in enumerate(frame_bounds[i:end]) if bounds is not None]
			if len(lowl) < 2 or len(highl) < 3:
				continue
			return lowl+highl
		return None

	out_bounds = [None for _ in frame_bounds]
	for i, bounds in enumerate(frame_bounds):
		subl = get_window(i)
		if subl is None:
			continue
		xp = [t[0] for t in subl]
		bounds = [[None, None] for _ in range(4)]
		for p_idx in range(4):
			for c_idx in range(2):
				yp = [t[1][p_idx][c_idx] for t in subl]
				coef = numpy.polyfit(xp, yp, 1)
				bounds[p_idx][c_idx] = int(numpy.poly1d(coef)(i))
		out_bounds[i] = bounds
	return out_bounds

video_path = sys.argv[1]
data_path = sys.argv[2]
scale = int(sys.argv[3])
cfg = Config(video_path, data_path)

print('reading bounds')
with open(os.path.join(cfg.data_path, 'align-gps.json'), 'r') as f:
	frame_bounds = json.load(f)

filter_bounds(frame_bounds)
frame_bounds = smooth2(frame_bounds)

print('reading frames')
t0 = time.time()
frame_ims = []
fnames = os.listdir(cfg.video_path)
fnames.sort()
for fname in fnames:
	if '.jpg' not in fname:
		continue
	frame_idx = int(fname.split('.jpg')[0])
	print('... {}/{}'.format(frame_idx, len(fnames)))

	im = skimage.io.imread(os.path.join(cfg.video_path, fname))
	im = cv2.resize(im, (im.shape[1]//scale, im.shape[0]//scale))

	while len(frame_ims) <= frame_idx:
		frame_ims.append(None)
	frame_ims[frame_idx] = im

t1 = time.time()
sift_features = sift.compute_sift(frame_ims)
t2 = time.time()
final_kps_all = find_kp.find_keypoints(cfg, frame_ims, sift_features, frame_bounds)
final_kps = get_good3(final_kps_all)
t3 = time.time()
out_bounds = align_homography.align(frame_ims, sift_features, frame_bounds, final_kps)
t4 = time.time()
filter_bounds(out_bounds)

with open(os.path.join(cfg.data_path, 'align-out.json'), 'w') as f:
	json.dump(out_bounds, f)

t5 = time.time()
print('read', t1-t0)
print('sift', t2-t1)
print('kps', t3-t2)
print('align', t4-t3)
print('nothing', t5-t4)

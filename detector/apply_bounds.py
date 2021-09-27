import cv2
import json
import numpy
import sys

bounds_fname = sys.argv[1]
raw_fname = sys.argv[2]
out_fname = sys.argv[3]

# Transform detections in raw_fname into detections in world coordinates.
# Do this based on the alignment data stored in bounds_fname.

with open(bounds_fname, 'r') as f:
	frame_bounds = json.load(f)

with open(raw_fname, 'r') as f:
	detections = json.load(f)

width = 1920
height = 1080
out_detections = [[] for _ in detections]

for frame_idx, dlist in enumerate(detections):
	bounds = frame_bounds[frame_idx]
	if bounds is None:
		continue

	src_points = numpy.array([
		[0, 0],
		[width, 0],
		[width, height],
		[0, height],
	], dtype='float32').reshape(-1, 1, 2)
	dst_points = numpy.array(bounds, dtype='float32').reshape(-1, 1, 2)
	H, _ = cv2.findHomography(src_points, dst_points)

	# transform detections
	points = []
	for d in dlist:
		points.extend(d['Points'])

	if len(points) > 0:
		points = numpy.array(points, dtype='float32').reshape(-1, 1, 2)
		try:
			transformed_points = cv2.perspectiveTransform(points, H)
		except Exception as e:
			print('... warning: perspectiveTransform error: {}'.format(e))
			continue
		i = 0
		out_dlist = []
		for orig_d in dlist:
			num_points = len(orig_d['Points'])
			cur_points = transformed_points[i:i+num_points, 0, :]
			i += num_points
			cur_points = [[int(x), int(y)] for x, y in cur_points]
			out_dlist.append({
				'Points': cur_points,
				'OrigPoints': orig_d['Points'],
			})
		assert i == transformed_points.shape[0]
		out_detections[frame_idx] = out_dlist

with open(out_fname, 'w') as f:
	json.dump(out_detections, f)

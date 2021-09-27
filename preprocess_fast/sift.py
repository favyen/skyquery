import cv2

sift = cv2.xfeatures2d.SIFT_create()

def compute_one(im):
	return sift.detectAndCompute(im, None)

def compute_sift(frames):
	print('get sift features')
	sift_features = [(None, None) for _ in frames]

	for frame_idx, im in enumerate(frames):
		if im is None or frame_idx % 3 != 0:
			continue

		print('... sift {}/{}'.format(frame_idx, len(frames)))
		keypoints, descs = compute_one(im)
		sift_features[frame_idx] = (keypoints, descs)

	return sift_features

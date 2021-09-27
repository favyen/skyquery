import model as model

import json
import numpy
import os, os.path
import skimage.io, skimage.transform
import sys
import tensorflow as tf

model_path = sys.argv[1]
video_path = sys.argv[2]
out_fname = sys.argv[3]

print('initializing model')
m = model.Model(size=(1080, 1080), bn=True)
session = tf.Session()
m.saver.restore(session, model_path)

def resize4(im):
	return skimage.transform.resize(im, [im.shape[0]*4, im.shape[1]*4], order=0, preserve_range=True).astype('uint8')

detections = []

for fname in os.listdir(video_path):
	if '.prev.jpg' not in fname:
		continue
	print(fname)
	frame_idx = int(fname.split('.')[0])
	im1 = skimage.io.imread(os.path.join(video_path, fname))
	im2 = skimage.io.imread(os.path.join(video_path, fname.replace('.prev.jpg', '.next.jpg')))
	cat_im = numpy.concatenate([im1, im2], axis=2)
	output1 = session.run(m.outputs, feed_dict={
		m.is_training: False,
		m.inputs: [cat_im[:, 0:1080, :]],
	})[0, :, :]
	output2 = session.run(m.outputs, feed_dict={
		m.is_training: False,
		m.inputs: [cat_im[:, 840:1920, :]],
	})[0, :, :]
	output = numpy.concatenate([output1[:, 0:240], output2[:, 30:270]], axis=1)

	if False:
		skimage.io.imsave('/home/ubuntu/vis/{}.a.jpg'.format(frame_idx), im1)
		skimage.io.imsave('/home/ubuntu/vis/{}.b.jpg'.format(frame_idx), im2)
		skimage.io.imsave('/home/ubuntu/vis/{}.o.png'.format(frame_idx), resize4(output*255))
		skimage.io.imsave('/home/ubuntu/vis/{}.ob.png'.format(frame_idx), resize4((output>0.1).astype('uint8')*255))

	while len(detections) <= frame_idx:
		detections.append([])

	while numpy.max(output) > 0.3:
		pos = numpy.unravel_index(numpy.argmax(output), output.shape)
		sx, sy, ex, ey = pos[1]-6, pos[0]-6, pos[1]+6, pos[0]+6
		if sx < 0:
			sx = 0
		if sy < 0:
			sy = 0
		if ex > output.shape[1]:
			ex = output.shape[1]
		if ey > output.shape[0]:
			ey = output.shape[0]
		output[sy:ey, sx:ex] = 0
		cx, cy = int(pos[1]), int(pos[0])
		detections[frame_idx].append({
			'Points': [
				[cx*4-20, cy*4-20],
				[cx*4+20, cy*4-20],
				[cx*4+20, cy*4+20],
				[cx*4-20, cy*4+20],
			],
		})

with open(out_fname, 'w') as f:
	json.dump(detections, f)

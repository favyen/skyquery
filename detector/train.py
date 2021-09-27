import model

import json
import numpy
import os
import random
import skimage.io, skimage.transform
import sys
import tensorflow as tf
import time

model_path = sys.argv[1]
resize = float(sys.argv[2])
data_paths = sys.argv[3:]

SIZE = 512

SIZE = int(SIZE*resize)
PADDING = int(6*resize)

print('reading frames')
train_tiles = []
val_tiles = []
for data_path in data_paths:
	for fname in os.listdir(data_path):
		if '.txt' not in fname:
			continue
		label = fname.split('.txt')[0]
		with open(data_path + label + '.txt') as f:
			positions = json.load(f)
		if len(positions) == 0:
			continue
		im1 = skimage.io.imread(data_path + label + '.a.jpg')
		im2 = skimage.io.imread(data_path + label + '.b.jpg')
		positions = [(p[1]*2, p[0]*2) for p in positions]

		if resize is not None and resize != 1:
			height, width = int(im1.shape[0]*resize), int(im1.shape[1]*resize)
			im1 = skimage.transform.resize(im1, (height, width), preserve_range=True).astype('uint8')
			im2 = skimage.transform.resize(im2, (height, width), preserve_range=True).astype('uint8')
			positions = [(int(p[0]*resize), int(p[1]*resize)) for p in positions]

		if int(label) >= 400 and int(label) < 440:
			val_tiles.append((im1, im2, positions))
		else:
			train_tiles.append((im1, im2, positions))

random.shuffle(train_tiles)
random.shuffle(val_tiles)

def extract(tiles):
	while True:
		im1, im2, positions = random.choice(tiles)
		i = random.randint(0, im1.shape[0]-SIZE)
		j = random.randint(0, im2.shape[1]-SIZE)
		pos_crop = [(p[0]-i, p[1]-j) for p in positions if p[0] > i and p[0] < i+SIZE and p[1] > j and p[1] < j+SIZE]
		if len(pos_crop) == 0:
			continue

		crop1 = im1[i:i+SIZE, j:j+SIZE, :]
		inoise = i + random.randint(int(-8*resize), int(8*resize))
		jnoise = j + random.randint(int(-8*resize), int(8*resize))
		if inoise < 0:
			inoise = 0
		elif inoise > im1.shape[0]-SIZE:
			inoise = im1.shape[0]-SIZE
		if jnoise < 0:
			jnoise = 0
		elif jnoise > im1.shape[1]-SIZE:
			jnoise = im1.shape[1]-SIZE
		crop2 = im2[inoise:inoise+SIZE, jnoise:jnoise+SIZE, :]

		if random.random() < 0.5:
			# vertical flip
			pos_crop = [(SIZE-p[0]-1, p[1]) for p in pos_crop]
			crop1 = numpy.flip(crop1, axis=0)
			crop2 = numpy.flip(crop2, axis=0)
		if random.random() < 0.5:
			# horizontal flip
			pos_crop = [(p[0], SIZE-p[1]-1) for p in pos_crop]
			crop1 = numpy.flip(crop1, axis=1)
			crop2 = numpy.flip(crop2, axis=1)

		return numpy.concatenate([crop1, crop2], axis=2), pos_crop

def get_gt_and_mask(output, positions):
	gt = numpy.zeros(output.shape, dtype='float32')
	mask = numpy.ones(output.shape, dtype='float32')
	for pos in positions:
		sx, sy, ex, ey = pos[0]//4-PADDING, pos[1]//4-PADDING, pos[0]//4+PADDING, pos[1]//4+PADDING
		if sx < 0:
			sx = 0
		if sy < 0:
			sy = 0
		if ex > SIZE//4:
			ex = SIZE//4
		if ey > SIZE//4:
			ey = SIZE//4
		mask[sx:ex, sy:ey] = 0
		max_indices = numpy.argmax(output[sx:ex, sy:ey])
		max_indices = numpy.unravel_index(max_indices, (ex-sx, ey-sy))
		bx, by = sx+max_indices[0], sy+max_indices[1]
		gt[bx, by] = 1
	mask = numpy.maximum(mask, gt)
	return gt, mask

val_examples = [extract(val_tiles) for _ in range(1024)]

print('initializing model')
m = model.Model(size=(SIZE, SIZE), bn=True)
session = tf.Session()
session.run(m.init_op)

print('begin training')
best_loss = None
no_improvement_epochs = 0
learning_rate = 1e-3

for epoch in range(9999):
	start_time = time.time()
	train_losses = []
	for _ in range(1024):
		batch_examples = [extract(train_tiles) for _ in range(model.BATCH_SIZE)]
		outputs = session.run(m.outputs, feed_dict={
			m.is_training: False,
			m.inputs: [example[0] for example in batch_examples],
		})
		gtmasks = [get_gt_and_mask(outputs[i, :, :], batch_examples[i][1]) for i in range(model.BATCH_SIZE)]
		_, loss = session.run([m.optimizer, m.loss], feed_dict={
			m.is_training: True,
			m.inputs: [example[0] for example in batch_examples],
			m.targets: [t[0] for t in gtmasks],
			m.masks: [t[1] for t in gtmasks],
			m.learning_rate: learning_rate,
		})
		train_losses.append(loss)
	train_loss = numpy.mean(train_losses)
	train_time = time.time()

	val_losses = []
	for i in range(0, len(val_examples), model.BATCH_SIZE):
		batch_examples = val_examples[i:i+model.BATCH_SIZE]
		outputs = session.run(m.outputs, feed_dict={
			m.is_training: False,
			m.inputs: [example[0] for example in batch_examples],
		})
		gtmasks = [get_gt_and_mask(outputs[i, :, :], batch_examples[i][1]) for i in range(model.BATCH_SIZE)]
		loss = session.run(m.loss, feed_dict={
			m.is_training: False,
			m.inputs: [example[0] for example in batch_examples],
			m.targets: [t[0] for t in gtmasks],
			m.masks: [t[1] for t in gtmasks],
		})
		val_losses.append(loss)

	val_loss = numpy.mean(val_losses)
	val_time = time.time()

	print('iteration {}: train_time={}, val_time={}, train_loss={}, val_loss={}/{}'.format(epoch, int(train_time - start_time), int(val_time - train_time), train_loss, val_loss, best_loss))

	if best_loss is None or val_loss < best_loss:
		best_loss = val_loss
		m.saver.save(session, model_path)
		no_improvement_epochs = 0
	else:
		no_improvement_epochs += 1
		if no_improvement_epochs >= 10 and learning_rate > 1e-4:
			print('drop learning rate to 1e-4')
			learning_rate = 1e-4
			no_improvement_epochs = 0
		elif no_improvement_epochs >= 20:
			break

def resize4(im):
	return skimage.transform.resize(im, [im.shape[0]*4, im.shape[1]*4], order=0, preserve_range=True).astype('uint8')

def test():
	for i, example in enumerate(val_examples[0:64]):
		output = session.run(m.outputs, feed_dict={
			m.is_training: False,
			m.inputs: [example[0]],
		})[0, :, :]
		gtim = numpy.zeros((SIZE//4, SIZE//4), dtype='uint8')
		for p in example[1]:
			gtim[p[0]//4, p[1]/4] = 255
		skimage.io.imsave('/home/ubuntu/vis/{}.a.jpg'.format(i), example[0][:, :, 0:3])
		skimage.io.imsave('/home/ubuntu/vis/{}.b.jpg'.format(i), example[0][:, :, 3:6])
		skimage.io.imsave('/home/ubuntu/vis/{}.gt.png'.format(i), resize4(gtim))
		skimage.io.imsave('/home/ubuntu/vis/{}.pos.out.png'.format(i), resize4((output*255).astype('uint8')))
		skimage.io.imsave('/home/ubuntu/vis/{}.pos.bin.png'.format(i), resize4((output>0.1).astype('uint8')*255))

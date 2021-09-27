from discoverlib import geom
import json
import os
import os.path
import random
import skimage.io

out_path = sys.argv[1]
resize = float(sys.argv[2])
data_paths = sys.argv[3:]

SIZE = 512

SIZE = int(SIZE*resize)
PADDING = int(6*resize)

# Assume inputs are 1920x1080.
crop_rects = [
	geom.Rectangle(geom.Point(0, 0), geom.Point(1024, 1024)),
	geom.Rectangle(geom.Point(896, 0), geom.Point(1920, 1024)),
]

counter = 0
train_set = []
val_set = []

for data_path in data_paths:
	labels = [fname.split('.')[0] for fname in os.listdir(data_path) if fname.endswith('.a.jpg')]
	for label in labels:
		txt_fname = ds_dir + label + '.txt'
		if not os.path.exists(txt_fname):
			continue
		with open(txt_fname, 'r') as f:
			points = json.load(f)
		if len(points) == 0:
			continue
		objects = [geom.Point(2*x, 2*y).bounds().add_tol(24) for x, y in points]
		im = skimage.io.imread(ds_dir + label + '.a.jpg')
		im2 = skimage.io.imread(ds_dir + label + '.b.jpg')
		for crop_rect in crop_rects:
			crop_len = crop_rect.lengths()
			crop_objects = []
			for obj_rect in objects:
				if not crop_rect.intersects(obj_rect):
					continue
				obj_clipped = crop_rect.clip_rect(obj_rect)
				if obj_clipped.start.x == obj_clipped.end.x or obj_clipped.start.y == obj_clipped.end.y:
					continue
				elif obj_clipped.lengths().x < 48 or obj_clipped.lengths().y < 48:
					continue
				obj_rel = geom.Rectangle(obj_clipped.start.sub(crop_rect.start), obj_clipped.end.sub(crop_rect.start))
				start = geom.FPoint(float(obj_rel.start.x) / crop_len.x, float(obj_rel.start.y) / crop_len.y)
				end = geom.FPoint(float(obj_rel.end.x) / crop_len.x, float(obj_rel.end.y) / crop_len.y)
				crop_objects.append((start.add(end).scale(0.5), end.sub(start)))

			if len(crop_objects) == 0:
				continue

			crop_im = im[crop_rect.start.y:crop_rect.end.y, crop_rect.start.x:crop_rect.end.x, :]
			crop_im2 = im2[crop_rect.start.y:crop_rect.end.y, crop_rect.start.x:crop_rect.end.x, :]
			crop_path = os.path.join(out_path, 'images/{}.jpg'.format(counter))
			skimage.io.imsave(crop_path, crop_im)
			skimage.io.imsave(crop_path.replace('.jpg', '.b.jpg'), crop_im2)
			crop_lines = ['0 {} {} {} {}'.format(center.x, center.y, size.x, size.y) for center, size in crop_objects]
			with open(os.path.join(out_path, 'images/{}.txt'.format(counter)), 'w') as f:
				f.write("\n".join(crop_lines) + "\n")

			if int(label) >= 400 and int(label) < 440:
				val_set.append(crop_path)
			else:
				train_set.append(crop_path)

			counter += 1

with open(os.path.join(out_path, 'train.txt'), 'w') as f:
	f.write("\n".join(train_set) + "\n")
with open(os.path.join(out_path, 'test.txt'), 'w') as f:
	f.write("\n".join(val_set) + "\n")
with open(os.path.join(out_path, 'valid.txt'), 'w') as f:
	f.write("\n".join(val_set) + "\n")

# convert geotiffs to pngs

import libtiff
import numpy
import skimage.io, skimage.transform
import sys

im_fname = sys.argv[1]
surface_fname = sys.argv[2]
out_fname = sys.argv[3]
masked_fname = sys.argv[4]
surface_thresholds = sys.argv[5]

#surface_thresholds = '-55,-32' for example data
threshold_parts = surface_thresholds.split(',')
surface_thresholds = [
    int(threshold_parts[0]),
    int(threshold_parts[1]),
]

im = libtiff.TIFF.open(sys.argv[1]).read_image()[:, :, 0:3]
surface = libtiff.TIFF.open(sys.argv[2]).read_image()
surface = numpy.logical_and(surface > surface_thresholds[0], surface < surface_thresholds[1])
surface = surface.astype('uint8')*255
surface = skimage.transform.resize(surface, (im.shape[0], im.shape[1]), order=0, preserve_range=True).astype('uint8')
skimage.io.imsave(sys.argv[3], im)
im = numpy.minimum(im, numpy.stack([surface, surface, surface], axis=2))
skimage.io.imsave(sys.argv[4], im)

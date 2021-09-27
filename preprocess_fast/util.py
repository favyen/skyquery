from discoverlib import geom

import math

def pad6(x):
	s = str(x)
	while len(s) < 6:
		s = '0' + s
	return s

def p_add(p1, p2):
	return (p1[0]+p2[0], p1[1]+p2[1])

def p_sub(p1, p2):
	return (p1[0]-p2[0], p1[1]-p2[1])

def p_scale(p, f):
	return (int(p[0]*f), int(p[1]*f))

def p_angle(p):
	return math.atan2(p[1], p[0])

def get_orientation(bounds):
	# get the two vectors corresponding to axes in bounds passing through center
	# then compute counterclockwise angle from positive x axis
	topleft, topright, bottomleft, bottomright = bounds
	top = p_scale(p_add(topleft, topright), 0.5)
	bottom = p_scale(p_add(bottomleft, bottomright), 0.5)
	left = p_scale(p_add(topleft, bottomleft), 0.5)
	right = p_scale(p_add(topright, bottomright), 0.5)
	v1 = p_sub(right, left)
	v2 = p_sub(top, bottom)
	a1 = p_angle(v1) - 0
	a2 = p_angle(v2) - something

def eucl(p1, p2):
	dx = p1[0] - p2[0]
	dy = p1[1] - p2[1]
	return math.sqrt(dx*dx+dy*dy)

def bounds_to_rect(bounds):
	rect = geom.Point(bounds[0][0], bounds[0][1]).bounds()
	rect = rect.extend(geom.Point(bounds[1][0], bounds[1][1]))
	rect = rect.extend(geom.Point(bounds[2][0], bounds[2][1]))
	rect = rect.extend(geom.Point(bounds[3][0], bounds[3][1]))
	return rect

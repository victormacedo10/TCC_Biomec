import time
import numpy as np
import math

def angle3pt(a, b, c):
    if b in (a, c):
        raise ValueError("Undefined angle, two identical points", (a, b, c))
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    if ang < 0:
        ang += 360
    if ang > 180:
        ang = 360 - ang
    return ang

def rectangularArea(person):
    max_x = max(person[:, 1])
    min_x = min([n for n in person[:, 1] if n>0])
    max_y = max(person[:, 0])
    min_y = min([n for n in person[:, 0] if n>0])
    return (max_x - min_x)*(max_y - min_y)
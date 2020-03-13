import numpy as np
import math

from .utils import fix

def rotate_vec(x, y, a):
    return x * math.cos(a) - y * math.sin(a), x * math.sin(a) + y * math.cos(a)

def rotate(x, y, origin_width, origin_height, new_width, new_height):
    x = x - new_width / 2
    y = y - new_height / 2

    length = math.sqrt(x * x + y * y)
    rate = length / math.sqrt(new_width * new_width + new_height * new_height)
    tx, ty = rotate_vec(x, y, math.pi * (15 * rate**2.5))

    delta = 1.0

    return fix(origin_width / 2 + tx * delta, origin_height / 2 + ty * delta, origin_width, origin_height)

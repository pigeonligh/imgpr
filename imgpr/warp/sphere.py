import numpy as np
import math

from .utils import fix

def sphere(x, y, origin_width, origin_height, new_width, new_height):
    x = x - new_width / 2
    y = y - new_height / 2

    p0 = 0.5 * min(new_width, new_height)
    d0 = 0.5 * max(origin_width, origin_height)

    p = math.sqrt(x * x + y * y)

    if p > p0:
        return -1, -1, True

    th = math.atan2(y + 1e-5, x + 1e-5)
    ph = math.asin(p / p0)

    d = 2 / math.pi * d0 * ph

    tx = d * math.cos(th) + origin_width / 2
    ty = d * math.sin(th) + origin_height / 2

    return fix(tx, ty, origin_width, origin_height)

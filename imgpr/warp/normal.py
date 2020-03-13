import numpy as np
from .utils import fix

def square(x, y, width, height, new_width, new_height):
    return fix(x, y * height / width, width, height)

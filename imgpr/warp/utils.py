def fix(x, y, origin_width, origin_height):
    flag = False
    if x < 0:
        x = 0
        flag = True
    if x > origin_width - 1:
        x = origin_width - 1
        flag = True
    if y < 0:
        y = 0
        flag = True
    if y > origin_height - 1:
        y = origin_height - 1
        flag = True
    return x, y, flag
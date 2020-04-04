import imgpr as ip
import numpy as np

image = ip.image.openImage("example.png")

height, width = image.shape[:2]
cut = 150

init_energy = np.zeros(shape=(height, width), dtype=int)
init_energy[520:,:] -= 14

x = ip.placeholder(shape=(height, width))
s = ip.layers.seam(x, iters=cut, init_energy=init_energy, direction=ip.VERTICAL)
y = ip.layers.sew(x, s, cut, shape=(height - cut, width), direction=ip.VERTICAL)

with ip.Session() as sess:
    output = sess.run(y, feed_dict={x : image})

ip.image.showImages([[image, output]])

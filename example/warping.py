import imgpr as ip

image = ip.image.openImage("example.png")

x = ip.placeholder(shape=image.shape[:2])
y = ip.layers.warping(x, (400, 400), ip.warp.sphere, fix_color=(200, 200, 200))

with ip.Session() as sess:
    output = sess.run(y, feed_dict={x : image})

ip.image.showImages([[image, output]])

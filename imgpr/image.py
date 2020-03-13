from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def openImage(imagefile):
    img = Image.open(imagefile)
    img = img.convert("RGB")
    data = np.array(img.getdata())
    data = data.reshape((img.height, img.width, 3))
    return data

def toImage(image):
    img = Image.new("RGB", (image.shape[1], image.shape[0]))
    image = image.reshape((-1, 3))
    image = map(lambda arr: (arr[0], arr[1], arr[2]), image.tolist())
    img.putdata(list(image))
    return img

def showImages(images, figsize=(8, 6)):
    plt.figure(figsize=figsize)
    if images:
        shape_0 = len(images)
        shape_1 = max(list(map(len, images)))
        for i in range(shape_0):
            for j in range(shape_1):
                if j >= len(images[i]):
                    break
                plt.subplot(shape_0, shape_1, i * shape_1 + j + 1)
                plt.imshow(images[i][j])
                plt.axis('off')
    plt.show()

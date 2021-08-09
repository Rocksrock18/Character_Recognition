import PIL
from PIL import Image
import numpy as np
import DataReader


w, h = 28, 28
m = DataReader.get_mapping()
data = DataReader.get_images(10, h, w) # 112800 images in data set
for image in data:
    print("\nCharacter being shown: " + chr(m[image[0]]))
    img = Image.fromarray(np.array(image[1], dtype=np.uint8))
    img.show()
    input()

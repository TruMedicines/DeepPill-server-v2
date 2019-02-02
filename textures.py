import os
import skimage
import skimage.transform
import skimage.io
import skimage.util
import numpy

textures = []

dirs = os.listdir('Pixar 130 Library')
for dir in dirs:
    if 'DS_Store' not in dir:
        files = os.listdir(f'Pixar 130 Library/{dir}')
        for file in files:
            if '_Normal' not in file and '_Roughness' not in file and '.DS_Store' not in file:
                image = skimage.io.imread(f'Pixar 130 Library/{dir}/{file}')

                image = skimage.transform.resize(image, (128, 128))
                tiles = numpy.array([image, image, image, image, image, image, image, image, image])
                image = skimage.util.montage(tiles, multichannel=True)

                crops = 4
                cropSize = int(128 / crops)

                for x in range(crops):
                    for y in range(crops):
                        cropLeft = x * cropSize
                        cropRight = 256 + cropLeft

                        cropTop = y * cropSize
                        cropBottom = 256 + cropTop

                        cropped = image[cropLeft:cropRight, cropTop:cropBottom]

                        textures.append(cropped)


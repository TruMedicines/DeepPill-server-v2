import os
import skimage
import skimage.transform
import skimage.io
import skimage.util
import numpy
import pkg_resources

textures = []

dirs = pkg_resources.resource_listdir("pillmatch", 'background_images')
for dir in dirs:
    if 'DS_Store' not in dir:
        files = pkg_resources.resource_listdir("pillmatch", f'background_images/{dir}')
        for file in files:
            if '_Normal' not in file and '_Roughness' not in file and '.DS_Store' not in file:
                image = skimage.io.imread(pkg_resources.resource_filename("pillmatch", f'background_images/{dir}/{file}'))

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

pkg_resources.cleanup_resources()

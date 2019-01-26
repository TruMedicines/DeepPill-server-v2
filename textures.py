import os
import skimage
import skimage.transform
import skimage.io

textures = []

dirs = os.listdir('Pixar 130 Library')
for dir in dirs:
    if 'DS_Store' not in dir:
        files = os.listdir(f'Pixar 130 Library/{dir}')
        for file in files:
            if '_Normal' not in file and '_Roughness' not in file and '.DS_Store' not in file:
                image = skimage.io.imread(f'Pixar 130 Library/{dir}/{file}')

                image = skimage.transform.resize(image, (256, 256))
                textures.append(image)


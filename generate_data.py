import numpy as np
import skimage
import scipy
import scipy.stats
import scipy.misc
import sklearn
import sklearn.preprocessing
import matplotlib.pyplot as plt


def generate_perlin_noise_2d(shape, res):
    def f(t):
        return 6* t ** 5 - 15 * t ** 4 + 10 * t ** 3

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    # Gradients
    angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    g00 = gradients[0:-1, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[0:-1, 1:].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:, 1:].repeat(d[0], 0).repeat(d[1], 1)
    # Ramps
    n00 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1])) * g00, 2)
    n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, 2)
    # Interpolation
    t = f(grid)
    n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
    n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
    return np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)


def generate_fractal_noise_2d(shape, res, octaves=1, persistence=0.5):
    noise = np.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * generate_perlin_noise_2d(shape, (frequency * res[0], frequency * res[1]))
        frequency *= 2
        amplitude *= persistence
    return noise


def cropCircle(imageData):
    width = imageData.shape[0]
    height = imageData.shape[1]

    circleCenterX = int(width/2)
    circleCenterY = int(height/2)

    circleRadius = int(min(width, height)/2)

    mask = np.zeros((width, height, 3), dtype=np.uint8)

    rr, cc = skimage.draw.circle(circleCenterY, circleCenterX, circleRadius, shape=(width, height))
    mask[rr, cc, :] = 1

    out = imageData * mask
    white = 1.0 * (1.0 - mask)

    imageData = out + white
    return imageData

def generatePillImage():
    width = 256
    height = 256

    splotchPatternWidth = 384
    splotchPatternHeight = 384

    # np.random.seed()
    splotchMaskPattern = generate_perlin_noise_2d((splotchPatternWidth, splotchPatternHeight), (16, 16))
    splotchMaskPattern = sklearn.preprocessing.binarize(splotchMaskPattern, threshold=0.45, copy=False)
    splotchMask = np.zeros((width, height, 1))
    splotchMask[:, :, 0] = splotchMaskPattern[:width, :height]
    splotchMask = scipy.ndimage.filters.gaussian_filter(splotchMask, 0.6, order=0)

    backgroundTexture = np.zeros((width, height, 3))
    backgroundPattern = np.zeros((width, height, 1))
    backgroundPattern[:,:,0] = generate_perlin_noise_2d((width, height), (16, 16))
    backgroundColor1 = np.zeros((width, height, 3))
    backgroundColor2 = np.zeros((width, height, 3))
    backgroundColor1[:,:] = np.array((242,246,238))/256.0
    backgroundColor2[:,:] = np.array((242,250,230))/256.0
    backgroundTexture[:,:] = backgroundPattern * backgroundColor1 + (1.0 - backgroundPattern) * backgroundColor2

    backgroundPattern2 = np.zeros((width, height, 1))
    backgroundPattern2[:,:,0] = generate_perlin_noise_2d((width, height), (64, 64))
    backgroundTexture[:,:] += (backgroundPattern2 - 1.0) * 0.02

    splotchTexture = np.zeros((width, height, 3))
    splotchPattern = np.zeros((width, height, 1))
    splotchPattern[:,:,0] = generate_perlin_noise_2d((width, height), (16, 16))
    splotchColor1 = np.zeros((width, height, 3))
    splotchColor2 = np.zeros((width, height, 3))
    splotchColor1[:,:] = np.array((181,67,10))/256.0
    splotchColor2[:,:] = np.array((170,12,12))/256.0
    splotchTexture[:,:] = splotchPattern * splotchColor1 + (1.0 - splotchPattern) * splotchColor2

    imageData = splotchMask * splotchTexture + (1.0 - splotchMask) * backgroundTexture
    imageData[:,:] += np.random.normal(loc=0.0, scale=0.01, size=(width, height, 3))

    imageData = cropCircle(imageData)

    return imageData




for n in range(10):
    imageData = generatePillImage()
    plt.imshow(imageData, interpolation='lanczos')
    # plt.show()

    plt.savefig('example-' + str(n) + ".png")

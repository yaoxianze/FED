import matplotlib.pyplot as plt
import skimage

plt.figure(figsize=(15,12))
origin = skimage.io.imread("plane.jpg")
noisy = skimage.util.random_noise(origin, mode='gaussian', var=0.1)
# skimage.io.imsave("plane_noise.jpg",noisy)
plt.imsave("plane_noise.jpg",noisy)
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from einops import rearrange
import cv2


def uniform_random(left, right, size = None):
    rand_nums = (right - left) * np.random.random(size) + left

    return rand_nums


def random_poly(edge_num, center, radius_range):
    angles = uniform_random(0, 2 * np.pi, edge_num)
    angles = np.sort(angles)
    random_radius = uniform_random(radius_range[0], radius_range[1], edge_num)
    x = np.cos(angles) * random_radius
    y = np.sin(angles) * random_radius
    x = np.expand_dims(x, 1)
    y = np.expand_dims(y, 1)
    points = np.concatenate([x, y], axis = 1)
    points += np.array(center)
    points = np.round(points).astype(np.int32)

    return points


def draw_poly(height, width, channel, points):
    image = np.zeros((height, width, channel), dtype = np.uint8)
    image = cv2.fillPoly(image, [points], 255)

    return image
    

def GenerateSCMap_poly(s, height = 32, width = 32, threshold = 1 / 16):
    img = np.zeros((s, height, width, 1))
    for i in range(s):
        num_polys = np.random.randint(5) + 1
        #print(num_polys)
        random_numbers = np.random.random(num_polys)
        normalized_numbers = random_numbers / random_numbers.sum()
        polys = np.zeros((num_polys, height, width, 1), dtype = np.uint8)

        for j in range(num_polys):
            while np.sum(polys[j]) < 255 * height * width * threshold:
                pts = random_poly(np.random.randint(5) + 3, [np.random.randint(height), np.random.randint(width)], [np.random.randint(10) + 1, np.random.randint(10) + 1])
                polys[j] = draw_poly(height, width, 1, pts)

            img[i] += normalized_numbers[j] * polys[j]
    
    img = rearrange(img, 's h w 1 -> s h w')
    return img

if __name__ == "__main__":
    output = GenerateSCMap_poly(1)
    output = rearrange(output, 's h w -> s h w 1')
    plt.imshow(output[0], cmap='gray', vmin = 0, vmax = 255)
    plt.colorbar()
    plt.show()

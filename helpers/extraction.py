import matplotlib.pyplot as plt
from numpy import arange, ndarray
from skimage import color
from skimage.draw import circle_perimeter
from skimage.feature import canny
from skimage.io import imread
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.util import img_as_ubyte


def detect_hough_circles(image_path: str, hough_radii: ndarray, circle_number: int):
    # Load image and detect edges
    image = img_as_ubyte(imread(image_path, as_gray=True))
    edges = canny(image, sigma=3, low_threshold=50, high_threshold=100)

    # Detect two radii
    hough_res = hough_circle(edges, hough_radii)

    # Select the most prominent circles
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=circle_number)

    # Draw them
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
    image = color.gray2rgb(image)
    for center_y, center_x, radius in zip(cy, cx, radii):
        circy, circx = circle_perimeter(center_y, center_x, radius, shape=image.shape)
        image[circy, circx] = (220, 20, 20)

    ax.imshow(image, cmap=plt.cm.gray)
    plt.suptitle(f"Canny: s=3, lt=50, ht=100; Hough Radii: {hough_radii}; Circles: {circle_number}")
    image_name = image_path.split(sep="/")
    plt.savefig(f"results/hough-circles/{image_name[2]}-{image_name[4]}-{image_name[5]}.jpg")


detect_hough_circles(
    image_path="./dataset/Test/Images/n02113799-standard_poodle/n02113799_815.jpg",
    hough_radii=arange(10, 80, 2),
    circle_number=10,
)

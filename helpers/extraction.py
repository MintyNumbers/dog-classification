import os

import matplotlib.pyplot as plt
from numpy import ndarray  # , arange
from skimage import color
from skimage.draw import circle_perimeter
from skimage.feature import canny, hog
from skimage.io import imread
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.util import img_as_ubyte


def detect_hough_circles(image_path: str, hough_radii: ndarray, circle_number: int):
    # Load image and detect edges
    image = img_as_ubyte(imread(image_path, as_gray=True))
    edges = canny(image, sigma=2, low_threshold=10, high_threshold=100)

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
    plt.savefig(f"../results/hough-circles/{image_name[2]}-{image_name[4]}-{image_name[5]}.jpg")


"""detect_hough_circles(
    image_path="../dataset/Test/Images/n02113799-standard_poodle/n02113799_815.jpg",
    hough_radii=arange(10, 80, 2),
    circle_number=10,
)"""


def detect_edges(image_path, save_plot=True):
    # Load image and detect edges
    image = img_as_ubyte(imread(image_path, as_gray=True))
    edges = canny(image, sigma=2, low_threshold=10, high_threshold=100)

    # Draw the edges
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
    ax.imshow(edges, cmap="gray")

    # Save plot
    if save_plot:
        image_name = image_path.split(sep="/")
        output_dir = "../results/canny"
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/{image_name[2]}-{image_name[3]}-{image_name[4]}.jpg")
    else:
        plt.show()


"""# Fur detection: sigma=0.75, low_threshold=10, high_threshold=100
detect_edges(image_path="../dataset/Images/n02113799-standard_poodle/n02113799_1864.jpg", save_plot=True)
detect_edges(image_path="../dataset/Images/n02113978-Mexican_hairless/n02113978_386.jpg", save_plot=True)

# General Shape (does not always work well with canny...)
detect_edges(image_path="../dataset/Images/n02115641-dingo/n02115641_1215.jpg", save_plot=True)"""


# HOG
def my_hog(image_path, save_plot=True):
    # Bild aus einer JPG-Datei laden
    img = imread(image_path, as_gray=True)

    # HOG-Merkmale extrahieren und visualisieren
    hog_features, hog_image = hog(
        img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True, visualize=True
    )

    # Originalbild und HOG-Bild plotten
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Original Image")
    plt.imshow(img, cmap="gray")

    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("HOG Image")
    plt.imshow(hog_image, cmap="gray")

    # Save plot
    if save_plot:
        image_name = image_path.split(sep="/")
        output_dir = "../results/hog"
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/{image_name[2]}-{image_name[3]}-{image_name[4]}.jpg")
    else:
        plt.show()


"""my_hog(image_path="../dataset/Images/n02115641-dingo/n02115641_1215.jpg")"""

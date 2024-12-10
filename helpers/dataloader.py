from glob import glob
from os import listdir

from cv2 import COLOR_BGR2RGB, IMREAD_COLOR, copyMakeBorder, cvtColor, imread, resize
from numpy import array, ndarray


def load_dataset(dataset_image_paths: str, scale=(256, 256)) -> tuple[ndarray, ndarray]:
    x = []
    y = []

    dataset_species_paths = listdir(dataset_image_paths)
    dataset_species_paths.sort()
    for species in dataset_species_paths:
        # fmt:off
        if   species == "n02113799-standard_poodle":     one_hot_encoded_species = array([1, 0, 0, 0, 0]) # noqa: E701
        elif species == "n02113978-Mexican_hairless":    one_hot_encoded_species = array([0, 1, 0, 0, 0]) # noqa: E701
        elif species == "n02115641-dingo":               one_hot_encoded_species = array([0, 0, 1, 0, 0]) # noqa: E701
        elif species == "n02115913-dhole":               one_hot_encoded_species = array([0, 0, 0, 1, 0]) # noqa: E701
        elif species == "n02116738-African_hunting_dog": one_hot_encoded_species = array([0, 0, 0, 0, 1]) # noqa: E701
        else:                                            one_hot_encoded_species = array([0, 0, 0, 0, 0]) # noqa: E701
        # fmt: on

        image_paths: list[str] = glob(f"{dataset_image_paths}/{species}/*")
        image_paths.sort()
        for image_path in image_paths:
            original_image = imread(image_path, flags=IMREAD_COLOR)  # BGR colors, HWC format
            rgb_image = cvtColor(original_image, code=COLOR_BGR2RGB)  # RGB colors, HWC format

            # if image smaller than scale pad around all sidses
            if rgb_image.shape[0] <= scale[0] and rgb_image.shape[1] <= scale[1]:
                top_bottom_padding = (scale[0] - rgb_image.shape[0]) // 2
                top_bottom_padding = (scale[1] - rgb_image.shape[1]) // 2
                padded_image = copyMakeBorder(
                    src=rgb_image,
                    top=top_bottom_padding,
                    bottom=top_bottom_padding,
                    left=top_bottom_padding,
                    right=top_bottom_padding,
                    borderType=0,
                )
            # if image is too long, pad left and right to square, then resize
            elif rgb_image.shape[0] > rgb_image.shape[1]:
                top_bottom_padding = (rgb_image.shape[0] - rgb_image.shape[1]) // 2
                padded_image = copyMakeBorder(
                    src=rgb_image,
                    top=0,
                    bottom=0,
                    left=top_bottom_padding,
                    right=top_bottom_padding,
                    borderType=0,
                )
            # if image is too broad, pad top and bottom to square, then resize
            elif rgb_image.shape[0] < rgb_image.shape[1]:
                top_bottom_padding = (rgb_image.shape[1] - rgb_image.shape[0]) // 2
                padded_image = copyMakeBorder(
                    src=rgb_image,
                    top=top_bottom_padding,
                    bottom=top_bottom_padding,
                    left=0,
                    right=0,
                    borderType=0,
                )

            resized_padded_image = resize(padded_image, scale)
            x.append(resized_padded_image)
            y.append(one_hot_encoded_species)

    return array(x), array(y)

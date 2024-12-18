from glob import glob
from os import listdir

from numpy import array, ndarray, pad
from skimage.io import imread
from skimage.transform import resize


def load_dataset(dataset_image_paths: str, scale: tuple[int, int, int] = (256, 256, 3)) -> tuple[ndarray, ndarray]:
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

        elif species == "n02111889-Samoyed":             one_hot_encoded_species = array([1, 0, 0, 0, 0]) # noqa: E701
        elif species == "n02112137-chow":                one_hot_encoded_species = array([0, 1, 0, 0, 0]) # noqa: E701
        elif species == "n02112350-keeshond":            one_hot_encoded_species = array([0, 0, 1, 0, 0]) # noqa: E701
        elif species == "n02112706-Brabancon_griffon":   one_hot_encoded_species = array([0, 0, 0, 1, 0]) # noqa: E701
        elif species == "n02113186-Cardigan":            one_hot_encoded_species = array([0, 0, 0, 0, 1]) # noqa: E701

        else: continue # noqa: E701
        # fmt: on

        image_paths: list[str] = glob(f"{dataset_image_paths}/{species}/*")
        image_paths.sort()
        for image_path in image_paths:
            original_image: ndarray = imread(image_path)  # RGB colors, HWC format

            # If image smaller than scale pad around all sides
            if original_image.shape[0] <= scale[0] and original_image.shape[1] <= scale[1]:
                top_bottom_padding = (scale[0] - original_image.shape[0]) // 2
                left_right_padding = (scale[1] - original_image.shape[1]) // 2
                padded_image = pad(
                    array=original_image,
                    pad_width=(
                        (top_bottom_padding, top_bottom_padding),  # Height
                        (left_right_padding, left_right_padding),  # Width
                        (0, 0),  # Channel
                    ),
                    mode="constant",
                )
            # If image is too long, pad left and right to square, then resize
            elif original_image.shape[0] > original_image.shape[1]:
                left_right_padding = (original_image.shape[0] - original_image.shape[1]) // 2
                padded_image = pad(
                    array=original_image,
                    pad_width=((0, 0), (left_right_padding, left_right_padding), (0, 0)),  # HWC
                    mode="constant",
                )
            # If image is too broad, pad top and bottom to square, then resize
            elif original_image.shape[0] < original_image.shape[1]:
                top_bottom_padding = (original_image.shape[1] - original_image.shape[0]) // 2
                padded_image = pad(
                    array=original_image,
                    pad_width=((top_bottom_padding, top_bottom_padding), (0, 0), (0, 0)),  # HWC
                    mode="constant",
                )

            resized_padded_image = resize(image=padded_image, output_shape=scale)

            x.append(resized_padded_image)
            y.append(one_hot_encoded_species)

    return array(x), array(y)

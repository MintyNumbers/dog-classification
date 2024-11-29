from glob import glob
from os import listdir, makedirs
from shutil import copy


def train_test_split() -> None:
    """Helper function to split the downloaded Dog classification dataset into a train and test subset."""

    # Dog species list that will be classified by the SVM and Deep Learning Model
    # Only species with at least num_samples samples will be used.
    # The second integer is the number of train samples
    num_samples = 150, 112
    DOG_SPECIES: list[str] = [
        "n02116738-African_hunting_dog",
        "n02115913-dhole",
        "n02115641-dingo",
        "n02113978-Mexican_hairless",
        "n02113799-standard_poodle",
    ]

    for species in DOG_SPECIES:
        # Prepare Images
        makedirs(f"./dataset/Train/Images/{species}", exist_ok=False)
        makedirs(f"./dataset/Test/Images/{species}", exist_ok=False)
        image_paths: list[str] = glob(f"./dataset/Images/{species}/*")
        image_paths.sort()

        # Prepare Annotation
        makedirs(f"./dataset/Train/Annotation/{species}", exist_ok=False)
        makedirs(f"./dataset/Test/Annotation/{species}", exist_ok=False)
        annotation_paths: list[str] = glob(f"./dataset/Annotation/{species}/*")
        annotation_paths.sort()

        for i in range(len(image_paths)):
            if i < num_samples[0]:
                if i < num_samples[1]:
                    copy(image_paths[i], f"./dataset/Train/Images/{species}/")
                    copy(annotation_paths[i], f"./dataset/Train/Annotation/{species}/")
                else:
                    copy(image_paths[i], f"./dataset/Test/Images/{species}/")
                    copy(annotation_paths[i], f"./dataset/Test/Annotation/{species}")
            else:
                break

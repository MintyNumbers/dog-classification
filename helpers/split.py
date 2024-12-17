from glob import glob
from os import makedirs
from shutil import copy


def train_test_split() -> None:
    """Helper function to split the downloaded Dog classification dataset into a train and test subset."""

    def copy_split(train_dir: str, test_dir: str, breeds_list: list[str]):
        for breed in breeds_list:
            # Prepare Images
            makedirs(f"{train_dir}/Images/{breed}", exist_ok=False)
            makedirs(f"{test_dir}/Images/{breed}", exist_ok=False)
            image_paths: list[str] = glob(f"./dataset/Images/{breed}/*")
            image_paths.sort()

            # Prepare Annotations
            makedirs(f"{train_dir}/Annotation/{breed}", exist_ok=False)
            makedirs(f"{test_dir}/Annotation/{breed}", exist_ok=False)
            annotation_paths: list[str] = glob(f"./dataset/Annotation/{breed}/*")
            annotation_paths.sort()

            # Copy Images and Annotations to new directories
            for i in range(len(image_paths)):
                if i < NUM_SAMPLES[0]:
                    if i < NUM_SAMPLES[1]:
                        copy(image_paths[i], f"{train_dir}/Images/{breed}/")
                        copy(annotation_paths[i], f"{train_dir}/Annotation/{breed}/")
                    else:
                        copy(image_paths[i], f"{test_dir}/Images/{breed}/")
                        copy(annotation_paths[i], f"{test_dir}/Annotation/{breed}")
                else:
                    break

    # First value is the total number of images, second value is the number of train images
    NUM_SAMPLES = 150, 112

    # List of dog breeds that will be classified by the SVM and Deep Learning Model
    DOG_BREEDS: list[str] = [  # Breeds used for classic ML and DL initial training
        "n02116738-African_hunting_dog",
        "n02115913-dhole",
        "n02115641-dingo",
        "n02113978-Mexican_hairless",
        "n02113799-standard_poodle",
    ]
    TRANSFER_LEARNING_DOG_BREEDS: list[str] = [  # Breeds used for DL transfer learning
        "n02113186-Cardigan",
        "n02112706-Brabancon_griffon",
        "n02112350-keeshond",
        "n02112137-chow",
        "n02111889-Samoyed",
    ]

    copy_split(
        train_dir="./dataset/Train",
        test_dir="./dataset/Test",
        breeds_list=DOG_BREEDS,
    )
    copy_split(
        train_dir="./dataset/Transfer-Train",
        test_dir="./dataset/Transfer-Test",
        breeds_list=TRANSFER_LEARNING_DOG_BREEDS,
    )

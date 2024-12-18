from keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D, RandomFlip, RandomRotation
from keras.models import Sequential
from keras.regularizers import L2


# Model architecture
def init_new_model(regularization: float, input_shape: tuple[int, int, int] = (256, 256, 3)) -> Sequential:
    model = Sequential(
        [
            Input(input_shape),
            # Data augmentation (only applied on model.fit())
            RandomFlip("horizontal_and_vertical", seed=0),
            RandomRotation(0.2, seed=0),
            # Convolution
            # Conv2D(16, (3, 3), activation="relu", kernel_regularizer=L2(regularization)),
            # MaxPooling2D((2, 2)),
            Conv2D(32, (3, 3), activation="relu", kernel_regularizer=L2(regularization)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation="relu", kernel_regularizer=L2(regularization)),
            MaxPooling2D((2, 2)),
            # Classification
            Flatten(),
            Dense(64, activation="relu", kernel_regularizer=L2(regularization)),
            Dense(5, activation="softmax"),  # 5 classes
        ]
    )
    return model

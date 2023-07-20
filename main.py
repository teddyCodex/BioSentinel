import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

################## define model architecture  #########################

# Set the input shape based on your image dimensions
input_shape = (512, 512, 3)

# Set the number of classes in your dataset
num_classes = 36

# Create the base model using MobileNetV2
base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet")

# Freeze the weights of the base model
base_model.trainable = False

# Create a new model on top of the base model
model = Sequential(
    [
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation="relu"),
        Dropout(0.5),
        Dense(
            num_classes, activation="softmax"
        ),  # num_classes is the number of classes in your dataset
    ]
)

################## compile model  #########################

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

################## Create data generators to load and preprocess the training and testing datasets ##################

# Set the path to your training and testing dataset directories
train_data_dir = "../biosentinel_training_data/fruit_vegs_tdata/train"
validation_data_dir = "../biosentinel_training_data/fruit_vegs_tdata/validation"
test_data_dir = "../biosentinel_training_data/fruit_vegs_tdata/test"

# Set the batch size and target image size
batch_size = 32
target_size = (512, 512)

# Create data generators with data augmentation for the training dataset
train_data_generator = ImageDataGenerator(
    rescale=1.0 / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True
)
train_generator = train_data_generator.flow_from_directory(
    train_data_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode="categorical",
)

validation_data_generator = ImageDataGenerator(rescale=1.0 / 255)  # Modify as needed

validation_generator = validation_data_generator.flow_from_directory(
    validation_data_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False,
)

# Create a data generator without data augmentation for the testing dataset
test_data_generator = ImageDataGenerator(rescale=1.0 / 255)
test_generator = test_data_generator.flow_from_directory(
    test_data_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False,
)

################## model training  #########################

# Set the number of training epochs
epochs = 10

# Train the model
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
)

################## model testing  #########################

# Evaluate the model on the testing dataset
evaluation = model.evaluate(test_generator, steps=test_generator.samples // batch_size)

# Print the evaluation metrics
print("Test Loss:", evaluation[0])
print("Test Accuracy:", evaluation[1])

# Save the trained model
model.save("BioSentinel.h5")

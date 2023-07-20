# After training the model
model.save("path/to/save/model.h5")

from tensorflow.keras.models import load_model

# Load the saved model
model = load_model("path/to/save/model.h5")

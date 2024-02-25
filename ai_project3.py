import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Dense, LSTM, Embedding, Input, Dropout, add
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import pickle
import nltk
from nltk.tokenize import word_tokenize

# Download NLTK data
nltk.download("punkt")

# Load pre-trained ResNet model
resnet = ResNet50(include_top=True, weights="imagenet")

# Remove classification layer
resnet = Model(inputs=resnet.input, outputs=resnet.layers[-2].output)


# Function to preprocess images
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img


# Function to encode image into features
def encode_image(image):
    image = preprocess_image(image)
    feature_vector = resnet.predict(image)
    feature_vector = np.reshape(feature_vector, feature_vector.shape[1])
    return feature_vector


# Load tokenizer
with open("tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

# Maximum sequence length
max_length = 40

# Load model
embedding_size = 300
vocab_size = len(tokenizer.word_index)
units = 256
encoder_dim = 2048

# Image feature extractor model
inputs1 = Input(shape=(encoder_dim,))
fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(units, activation="relu")(fe1)

# Caption sequence model
inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, embedding_size, mask_zero=True)(inputs2)
se2 = Dropout(0.5)(se1)
se3 = LSTM(units)(se2)

# Merging both models
decoder1 = add([fe2, se3])
decoder2 = Dense(units, activation="relu")(decoder1)
outputs = Dense(vocab_size, activation="softmax")(decoder2)

# Combined model
model = Model(inputs=[inputs1, inputs2], outputs=outputs)

# Load trained weights
model.load_weights("model_weights.h5")


# Function to generate caption
def generate_caption(image_path):
    photo = encode_image(image_path)
    in_text = "<start>"
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo.reshape(1, -1), sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word[yhat]
        in_text += " " + word
        if word == "<end>":
            break
    caption = in_text.split()
    caption = caption[1:-1]
    caption = " ".join(caption)
    return caption


# Example usage
image_path = "example.jpg"
caption = generate_caption(image_path)
print("Generated Caption:", caption)

# Plot the image
img = plt.imread(image_path)
plt.imshow(img)
plt.title(caption)
plt.axis("off")
plt.show()

# Adding external dataset and captions
external_data = [
    ("external_image1.jpg", "A beautiful sunset over the mountains."),
    ("external_image2.jpg", "A cute dog playing in the park."),
    # Add more images and captions as needed
]

for image_path, caption in external_data:
    print("Image Path:", image_path)
    print("Caption:", caption)
    caption_generated = generate_caption(image_path)
    print("Generated Caption:", caption_generated)
    print()

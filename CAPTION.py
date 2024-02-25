import numpy as np
import matplotlib.pyplot as plt
import pickle

from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model, load_model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input, Dense, LSTM, Embedding, Dropout
from keras.preprocessing.text import Tokenizer

# Load pre-trained VGG16 model without the top layer
base_model = VGG16(weights="imagenet")
base_model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)


# Define a function to preprocess images
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


# Define a function to extract features from images using VGG16
def extract_features(img):
    return base_model.predict(img)


# Load or create tokenizer
def load_tokenizer(tokenizer_path):
    try:
        with open(tokenizer_path, "rb") as f:
            tokenizer = pickle.load(f)
    except:
        tokenizer = Tokenizer()
    return tokenizer


def create_tokenizer(captions):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(captions)
    return tokenizer


# Load or create tokenizer
tokenizer_path = "tokenizer.pkl"
captions = ["a cat is sitting on the table", "a dog is running in the park", ...]
tokenizer = load_tokenizer(tokenizer_path)
if tokenizer is None:
    tokenizer = create_tokenizer(captions)
    with open(tokenizer_path, "wb") as f:
        pickle.dump(tokenizer, f)

# Load pre-trained LSTM model for caption generation
caption_model = load_model("caption_generator_model.h5")  # Replace with your model path


# Define a function to generate captions for images
def generate_caption(img_path):
    img = preprocess_image(img_path)
    features = extract_features(img)
    start_token = "<start>"
    caption = [start_token]

    while True:
        sequence = tokenizer.texts_to_sequences([caption])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)

        preds = caption_model.predict([features, sequence], verbose=0)
        pred_word_index = np.argmax(preds)

        word = index_word[pred_word_index]

        if word is None or word == "<end>" or len(caption) >= max_length:
            break

        caption.append(word)

        if word == "<end>":
            break

    return " ".join(caption[1:-1])  # Exclude start and end tokens


# Test the image captioning function
img_path = "example_image.jpg"
caption = generate_caption(img_path)
print("Generated Caption:", caption)

# Display the image
img = image.load_img(img_path)
plt.imshow(img)
plt.axis("off")
plt.show()

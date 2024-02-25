import os
import cv2
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Dense, LSTM, Embedding, Input, Dropout, add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


# Function to preprocess images
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = preprocess_input(img)
    return img


# Read captions from text file
captions_file = "captions.txt"
captions = {}
with open(captions_file, "r") as file:
    for line in file:
        image_filename, caption = line.strip().split(",")
        captions[image_filename.strip()] = caption.strip()

# Prepare image data and corresponding captions
dataset_folder = "dataset"
images = []
captions_list = []
for image_filename in os.listdir(dataset_folder):
    image_path = os.path.join(dataset_folder, image_filename)
    image = preprocess_image(image_path)
    images.append(image)
    captions_list.append(captions[image_filename])

# Convert captions to sequences of token indices
tokenizer = Tokenizer()
tokenizer.fit_on_texts(captions_list)
captions_sequences = tokenizer.texts_to_sequences(captions_list)

# Define model architecture
image_input = Input(shape=(224, 224, 3))
resnet = ResNet50(include_top=False, weights="imagenet", input_tensor=image_input)
for layer in resnet.layers:
    layer.trainable = False

encoder_output = resnet.output

decoder_input = Input(shape=(None,))
embedding_layer = Embedding(
    input_dim=len(tokenizer.word_index) + 1, output_dim=300, mask_zero=True
)(decoder_input)
decoder_lstm = LSTM(256)(embedding_layer)

decoder_concat = add([encoder_output, decoder_lstm])
decoder_dense = Dense(len(tokenizer.word_index) + 1, activation="softmax")(
    decoder_concat
)

model = Model(inputs=[image_input, decoder_input], outputs=decoder_dense)
model.compile(loss="categorical_crossentropy", optimizer=Adam())

# Train model
images_array = np.array(images)
captions_sequences_padded = pad_sequences(captions_sequences, padding="post")
targets = to_categorical(
    captions_sequences_padded, num_classes=len(tokenizer.word_index) + 1
)

model.fit(
    [images_array, captions_sequences_padded],
    targets,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
)

# Save model weights
model.save_weights("model_weights.h5")

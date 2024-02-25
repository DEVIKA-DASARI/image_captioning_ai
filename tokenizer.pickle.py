import numpy as np
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer

# Example captions
captions = [
    "A girl is looking outside.",
    "A girl sitting on the chair.",
    "A girl and a boy are sitting at the beach side enjoying the night sky.",
]

# Initialize tokenizer
tokenizer = Tokenizer()

# Fit tokenizer on captions
tokenizer.fit_on_texts(captions)

# Save tokenizer to file
with open("tokenizer.pickle", "wb") as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print("Vocabulary Size:", vocab_size)

# Example usage to convert text to sequences
sequences = tokenizer.texts_to_sequences(captions)
print("Example Sequence:", sequences[0])

import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# 1. Setup Directories
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

print("Downloading massive dataset from Hugging Face...")
# We use a widely recognized open-source customer support dataset
dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
df = pd.DataFrame(dataset['train'])

# Keep only the relevant columns: what the user says (instruction) and the intent
df = df[['instruction', 'intent']]
df.dropna(inplace=True)
print(f"Dataset loaded with {len(df)} real-world queries!")

# 2. Preprocess Labels
print("Encoding labels...")
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['intent'])
num_classes = len(label_encoder.classes_)

# Save the encoder so the backend knows what the predictions mean
with open('models/label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

# Create a mapping of intents to automated responses (Rule-based fallback mapping)
response_mapping = {
    intent: f"I can help you with your {intent.replace('_', ' ')}. Let me connect you to the right workflow or provide the standard procedure."
    for intent in label_encoder.classes_
}
with open('models/response_mapping.pkl', 'wb') as f:
    pickle.dump(response_mapping, f)

# 3. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(df['instruction'], df['label'], test_size=0.2, random_state=42)

# 4. Text Vectorization (Converting words to numbers efficiently)
print("Building Text Vectorizer...")
VOCAB_SIZE = 10000
MAX_LENGTH = 50

vectorize_layer = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode='int',
    output_sequence_length=MAX_LENGTH
)
vectorize_layer.adapt(X_train.to_numpy())

# Save the vectorizer weights
pickle.dump({'config': vectorize_layer.get_config(),
             'weights': vectorize_layer.get_weights()}, 
            open('models/vectorizer.pkl', 'wb'))

# 5. Build the Deep Learning Model (Embedding Architecture)
print("Building the Neural Network...")
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,), dtype=tf.string),
    vectorize_layer,
    tf.keras.layers.Embedding(VOCAB_SIZE, 64, mask_zero=True),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 6. Train the Model
print("Starting Training (This may take a few minutes)...")
history = model.fit(
    X_train.to_numpy(), y_train.to_numpy(),
    validation_data=(X_test.to_numpy(), y_test.to_numpy()),
    epochs=15, 
    batch_size=32
)

# 7. Save the Model
model.save('models/support_ai_model.keras')
print("Production Model trained and saved successfully in /models!")

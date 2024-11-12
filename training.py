import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense

# Step 1: Load and Filter the Data
# Load the CSV file
data = pd.read_csv("filtered_data.csv", encoding="ISO-8859-1")

# Filter rows where 'q_type' ends with '2Jr'
filtered_data = data[data['q_type'].str.endswith("2Jr", na=False)]
filtered_data = filtered_data[filtered_data['j_race'].isin(['White', 'Black'])]
# Step 2: Prepare Text and Labels
# Get the content and labels, ensuring 'q_content' entries are strings
texts = filtered_data['q_content'].fillna("").astype(str).values
labels = filtered_data['j_race'].values
print(pd.Series(labels).unique())
# Encode the labels into numeric form (0, 1) for binary classification
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)  # e.g., "White" -> 0, "Black" -> 1
print("Unique values in labels:", pd.Series(labels).unique())
print("Number of NaNs in texts:", pd.Series(texts).isna().sum())
# Step 3: Tokenize and Pad the Text Data
max_words = 10000  # Vocabulary size
max_len = 100      # Max length of sequences

# Initialize and fit tokenizer
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)

# Convert texts to sequences and pad them
sequences = tokenizer.texts_to_sequences(texts)
x_data = pad_sequences(sequences, maxlen=max_len)

# Step 4: Split the Data into Training and Validation Sets
x_train, x_val, y_train, y_val = train_test_split(x_data, labels, test_size=0.2, random_state=42)

# Step 5: Define the CNN Model
model = Sequential([
    Embedding(input_dim=max_words, output_dim=50, input_length=max_len),
    Conv1D(128, 5, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(10, activation='relu'),
    Dense(1, activation='sigmoid')  # Sigmoid activation for binary classification
])

# Step 6: Compile the Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 7: Train the Model
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_val, y_val))

# Step 8: Evaluate the Model
loss, accuracy = model.evaluate(x_val, y_val)
print(f"Validation Accuracy: {accuracy}")

# Step 9: Save the Model (Optional)
model.save("juror_classification_model.h5")


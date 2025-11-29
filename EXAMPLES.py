"""
Example: Training and using embeddings for cross-domain recommendation research
"""

import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

# Example 1: Train both domains
print("="*60)
print("Example 1: Train Both Domains")
print("="*60)
print("""
# Train Books domain
python src/train_single_domain.py --domain books

# Train Movies domain  
python src/train_single_domain.py --domain movies
""")

# Example 2: Load embeddings for one domain
print("\n" + "="*60)
print("Example 2: Load and Use Embeddings")
print("="*60)
print("""
import numpy as np

# Load Books domain embeddings
data = np.load('outputs/books_TIMESTAMP/embeddings.npz', allow_pickle=True)

user_embeddings = data['user_embeddings']  # Shape: (10485, 64)
item_embeddings = data['item_embeddings']  # Shape: (8318, 64)
user2idx = data['user2idx'].item()  # Dict: user_id -> index
item2idx = data['item2idx'].item()  # Dict: item_id -> index

# Get embedding for a specific user
user_id = "A2S166WSCFIFP5"
user_idx = user2idx[user_id]
user_embedding = user_embeddings[user_idx]

print(f"User {user_id} embedding shape: {user_embedding.shape}")
print(f"First 10 dimensions: {user_embedding[:10]}")
""")

# Example 3: Extract overlapping users
print("\n" + "="*60)
print("Example 3: Extract Overlapping User Embeddings")
print("="*60)
print("""
from src.utils import extract_overlapping_embeddings

# Extract embeddings for users that appear in both domains
result = extract_overlapping_embeddings(
    books_embeddings_path='outputs/books_TIMESTAMP/embeddings.npz',
    movies_embeddings_path='outputs/movies_TIMESTAMP/embeddings.npz',
    output_path='outputs/overlapping_embeddings.npz'
)

books_emb = result['books_embeddings']    # Shape: (1062, 64)
movies_emb = result['movies_embeddings']  # Shape: (1062, 64)
user_ids = result['overlapping_user_ids'] # List of 1062 user IDs

print(f"Number of overlapping users: {len(user_ids)}")
print(f"Books embeddings shape: {books_emb.shape}")
print(f"Movies embeddings shape: {movies_emb.shape}")

# Now these paired embeddings can be used to learn a mapping:
# f: books_emb -> movies_emb
""")

# Example 4: Train a simple linear mapping (for cross-domain transfer)
print("\n" + "="*60)
print("Example 4: Learn a Cross-Domain Mapping")
print("="*60)
print("""
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# Load overlapping embeddings
data = np.load('outputs/overlapping_embeddings.npz', allow_pickle=True)
X = data['books_embeddings']   # Source domain (Books)
Y = data['movies_embeddings']  # Target domain (Movies)

# Split into train/test
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# Convert to tensors
X_train = torch.FloatTensor(X_train)
Y_train = torch.FloatTensor(Y_train)
X_test = torch.FloatTensor(X_test)
Y_test = torch.FloatTensor(Y_test)

# Define a simple linear mapping
class LinearMapping(nn.Module):
    def __init__(self, input_dim=64, output_dim=64):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)

# Train the mapping
model = LinearMapping()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

for epoch in range(100):
    optimizer.zero_grad()
    pred = model(X_train)
    loss = criterion(pred, Y_train)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Evaluate
model.eval()
with torch.no_grad():
    pred = model(X_test)
    test_loss = criterion(pred, Y_test)
    print(f"Test Loss: {test_loss.item():.4f}")

# Now you can use this model to transfer knowledge from Books to Movies!
# Given a user's Books embedding, predict their Movies embedding
""")

# Example 5: Make cross-domain recommendations
print("\n" + "="*60)
print("Example 5: Cross-Domain Recommendation")
print("="*60)
print("""
import torch

# Scenario: A user has interacted with Books, but not Movies
# We want to recommend Movies based on their Books preferences

# 1. Load both domain models
books_model = load_model('outputs/books_TIMESTAMP/model.pt')
movies_model = load_model('outputs/movies_TIMESTAMP/model.pt')

# 2. Load the learned mapping
mapping_model = load_trained_mapping('mapping_model.pt')

# 3. Get user's Books embedding
user_id = "NEW_USER_123"
books_user_idx = books_user2idx[user_id]
books_embedding = books_model.get_user_embedding(books_user_idx)

# 4. Map to Movies domain using learned mapping
predicted_movies_embedding = mapping_model(books_embedding)

# 5. Find similar users in Movies domain
movies_user_embeddings = movies_model.get_user_embeddings()
similarities = torch.matmul(
    predicted_movies_embedding,
    movies_user_embeddings.t()
)
most_similar_users = torch.topk(similarities, k=10)

# 6. Recommend items that similar users liked in Movies domain
# (Implement collaborative filtering logic here)

print("Generated cross-domain recommendations!")
""")

# Example 6: Evaluate cross-domain transfer
print("\n" + "="*60)
print("Example 6: Evaluate Cross-Domain Transfer")
print("="*60)
print("""
from sklearn.metrics import mean_squared_error
import numpy as np

# Load overlapping embeddings
data = np.load('outputs/overlapping_embeddings.npz', allow_pickle=True)
books_emb = data['books_embeddings']
movies_emb = data['movies_embeddings']

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(
    books_emb, movies_emb, test_size=0.2, random_state=42
)

# Baseline: Identity mapping (no transfer)
baseline_mse = mean_squared_error(X_test, Y_test)

# Your model: Learned mapping
Y_pred = mapping_model.predict(X_test)
model_mse = mean_squared_error(Y_test, Y_pred)

print(f"Baseline MSE (no transfer): {baseline_mse:.4f}")
print(f"Model MSE (learned mapping): {model_mse:.4f}")
print(f"Improvement: {(1 - model_mse/baseline_mse)*100:.2f}%")

# Compute cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

baseline_cos = np.mean([
    cosine_similarity(x, y) for x, y in zip(X_test, Y_test)
])

model_cos = np.mean([
    cosine_similarity(pred, y) for pred, y in zip(Y_pred, Y_test)
])

print(f"\\nBaseline cosine similarity: {baseline_cos:.4f}")
print(f"Model cosine similarity: {model_cos:.4f}")
""")

print("\n" + "="*60)
print("✓ Examples Complete!")
print("="*60)
print("\nThese examples show you how to:")
print("1. Train models on both domains")
print("2. Load and use embeddings")
print("3. Extract overlapping users")
print("4. Learn cross-domain mappings")
print("5. Make cross-domain recommendations")
print("6. Evaluate transfer performance")
print("\nYou can adapt these examples for your Wasserstein CDR research!")


"""
Guide: How Embeddings Are Stored

This document explains the structure and access patterns for saved embeddings.
"""

import numpy as np

# ==================================================================
# EMBEDDINGS FILE FORMAT
# ==================================================================

"""
The embeddings are saved in NumPy's compressed format (.npz) with 4 arrays:

outputs/books_TIMESTAMP/embeddings.npz contains:
├── user_embeddings    : numpy array (n_users × embedding_dim)
├── item_embeddings    : numpy array (n_items × embedding_dim)
├── user2idx          : dict {user_id_string → index}
└── item2idx          : dict {item_id_string → index}
"""

# ==================================================================
# HOW TO LOAD EMBEDDINGS
# ==================================================================

# Method 1: Load everything
data = np.load('outputs/books_TIMESTAMP/embeddings.npz', allow_pickle=True)

user_embeddings = data['user_embeddings']  # Shape: (10485, 64)
item_embeddings = data['item_embeddings']  # Shape: (8318, 64)
user2idx = data['user2idx'].item()  # Dict: user_id → index
item2idx = data['item2idx'].item()  # Dict: item_id → index

print("User embeddings shape:", user_embeddings.shape)
# Output: (10485, 64) = 10,485 users × 64 dimensions

print("Item embeddings shape:", item_embeddings.shape)
# Output: (8318, 64) = 8,318 items × 64 dimensions

# ==================================================================
# ACCESSING SPECIFIC USER EMBEDDINGS
# ==================================================================

# Example 1: Get embedding for a specific user
user_id = "A2S166WSCFIFP5"  # Original user ID from dataset

# Step 1: Look up the index
user_idx = user2idx[user_id]  # Returns integer index

# Step 2: Get the embedding
user_embedding = user_embeddings[user_idx]  # Shape: (64,)

print(f"User {user_id} embedding:")
print(f"  Index: {user_idx}")
print(f"  Shape: {user_embedding.shape}")
print(f"  First 10 dims: {user_embedding[:10]}")

# ==================================================================
# ACCESSING MULTIPLE USERS
# ==================================================================

# Example 2: Get embeddings for multiple users
user_ids = ["A2S166WSCFIFP5", "A2XQ5LZHTD4AFT", "A2I35JB67U20C0"]

# Get indices
indices = [user2idx[uid] for uid in user_ids]

# Get embeddings (vectorized)
batch_embeddings = user_embeddings[indices]  # Shape: (3, 64)

print(f"Batch embeddings shape: {batch_embeddings.shape}")

# ==================================================================
# FILE SIZE AND MEMORY
# ==================================================================

"""
Memory usage per embedding:
- Each embedding: 64 dimensions × 4 bytes (float32) = 256 bytes
- 10,485 users: 10,485 × 256 bytes ≈ 2.6 MB
- 8,318 items: 8,318 × 256 bytes ≈ 2.1 MB
- Total: ~5 MB for embeddings

The actual .npz file is ~5 MB due to compression.
"""

# ==================================================================
# STORAGE FORMAT DETAILS
# ==================================================================

"""
NumPy .npz format:
- Compressed archive (like a ZIP file)
- Contains multiple numpy arrays
- Efficient for large datasets
- Can be loaded partially (lazy loading)

Data types:
- user_embeddings: float32 (to save memory)
- item_embeddings: float32
- user2idx: Python dict (stored as numpy object)
- item2idx: Python dict (stored as numpy object)
"""

# ==================================================================
# PRACTICAL EXAMPLES
# ==================================================================

def load_embeddings_example():
    """Complete example of loading and using embeddings."""
    
    # Load
    data = np.load('outputs/books_20251128_234525/embeddings.npz', allow_pickle=True)
    user_embeddings = data['user_embeddings']
    user2idx = data['user2idx'].item()
    
    # Example 1: Get single user embedding
    user_id = "A2S166WSCFIFP5"
    if user_id in user2idx:
        idx = user2idx[user_id]
        emb = user_embeddings[idx]
        print(f"✓ Found user {user_id}")
        print(f"  Embedding: {emb[:5]}... (showing first 5 dims)")
    else:
        print(f"✗ User {user_id} not found")
    
    # Example 2: Get all users with embeddings
    all_user_ids = list(user2idx.keys())
    print(f"\n✓ Total users: {len(all_user_ids)}")
    print(f"  First 5: {all_user_ids[:5]}")
    
    # Example 3: Compute similarity between two users
    user1 = "A2S166WSCFIFP5"
    user2 = "A2XQ5LZHTD4AFT"
    
    emb1 = user_embeddings[user2idx[user1]]
    emb2 = user_embeddings[user2idx[user2]]
    
    # Cosine similarity
    cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    print(f"\n✓ Similarity between {user1} and {user2}: {cos_sim:.4f}")
    
    # Example 4: Find most similar users (simple version)
    target_user = "A2S166WSCFIFP5"
    target_emb = user_embeddings[user2idx[target_user]]
    
    # Compute similarity with all users
    similarities = user_embeddings @ target_emb  # Dot product
    similarities = similarities / (np.linalg.norm(user_embeddings, axis=1) * np.linalg.norm(target_emb))
    
    # Get top 5 most similar (excluding self)
    top_indices = np.argsort(similarities)[-6:-1][::-1]  # Top 5, excluding self
    
    print(f"\n✓ Most similar users to {target_user}:")
    idx2user = {v: k for k, v in user2idx.items()}
    for rank, idx in enumerate(top_indices, 1):
        similar_user = idx2user[idx]
        sim_score = similarities[idx]
        print(f"  {rank}. {similar_user} (similarity: {sim_score:.4f})")

# ==================================================================
# CROSS-DOMAIN USAGE
# ==================================================================

def cross_domain_example():
    """Example of using embeddings from both domains."""
    
    # Load Books domain
    books = np.load('outputs/books_TIMESTAMP/embeddings.npz', allow_pickle=True)
    books_user_emb = books['user_embeddings']
    books_user2idx = books['user2idx'].item()
    
    # Load Movies domain
    movies = np.load('outputs/movies_TIMESTAMP/embeddings.npz', allow_pickle=True)
    movies_user_emb = movies['user_embeddings']
    movies_user2idx = movies['user2idx'].item()
    
    # Find overlapping users
    books_users = set(books_user2idx.keys())
    movies_users = set(movies_user2idx.keys())
    overlap = books_users & movies_users
    
    print(f"✓ Overlapping users: {len(overlap)}")
    
    # Get paired embeddings for overlapping users
    paired_books = []
    paired_movies = []
    
    for user_id in overlap:
        books_idx = books_user2idx[user_id]
        movies_idx = movies_user2idx[user_id]
        
        paired_books.append(books_user_emb[books_idx])
        paired_movies.append(movies_user_emb[movies_idx])
    
    paired_books = np.array(paired_books)  # Shape: (1062, 64)
    paired_movies = np.array(paired_movies)  # Shape: (1062, 64)
    
    print(f"✓ Paired Books embeddings: {paired_books.shape}")
    print(f"✓ Paired Movies embeddings: {paired_movies.shape}")
    
    # Now you can learn a mapping: f(paired_books) → paired_movies

# ==================================================================
# QUICK REFERENCE
# ==================================================================

"""
QUICK REFERENCE:

1. Load embeddings:
   data = np.load('path/to/embeddings.npz', allow_pickle=True)
   user_embeddings = data['user_embeddings']
   user2idx = data['user2idx'].item()

2. Get single user:
   idx = user2idx[user_id]
   embedding = user_embeddings[idx]

3. Get batch of users:
   indices = [user2idx[uid] for uid in user_ids]
   batch_emb = user_embeddings[indices]

4. File structure:
   - user_embeddings: (n_users, embedding_dim) numpy array
   - item_embeddings: (n_items, embedding_dim) numpy array
   - user2idx: {user_id: index} dictionary
   - item2idx: {item_id: index} dictionary

5. Memory: ~5 MB per domain (Books or Movies)
"""

if __name__ == "__main__":
    print(__doc__)
    print("\n" + "="*60)
    print("Running Examples")
    print("="*60)
    
    # Uncomment to run examples:
    # load_embeddings_example()
    # cross_domain_example()
    
    print("\n✓ See code above for complete examples!")


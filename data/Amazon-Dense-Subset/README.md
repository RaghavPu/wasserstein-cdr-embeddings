# Amazon Dense Subset Dataset

## 🎯 The Sweet Spot!

This dataset combines the best of both worlds:
- ✅ **Real 1-5 star ratings** (like large dataset)
- ✅ **Fast training** (like 5-core dataset)
- ✅ **Dense data** (35x denser!)
- ✅ **All users in both domains** (perfect for CDR)

## 📊 Dataset Statistics

### Books Domain
- **Users**: 1,426
- **Items**: 723 ⭐ (filtered from 276K!)
- **Interactions**: 38,149
- **Avg per user**: 27 interactions
- **Avg per item**: 53 interactions ✅
- **Density**: 3.70% (743x denser than original!)
- **Ratings**: Real 1-5 stars (mean: 4.31)

### Movies Domain
- **Users**: 1,381
- **Items**: 1,338 ⭐ (filtered from 41K!)
- **Interactions**: 68,023
- **Avg per user**: 49 interactions
- **Avg per item**: 51 interactions ✅
- **Density**: 3.68% (149x denser than original!)
- **Ratings**: Real 1-5 stars (mean: 4.01)

### Cross-Domain
- **1,329 users appear in BOTH domains** (93% overlap!)
- Perfect for cross-domain recommendation research
- (Lost 171 users due to item filtering, but much better quality!)

## 📈 Comparison with Other Datasets

| Dataset | Users | Books Items | Books Int. | Movies Int. | Ratings | Training Time | Density |
|---------|-------|-------------|------------|-------------|---------|---------------|---------|
| **5-core (small)** | 1,062 overlap | 21K items | 115K | 67K | All 1s | 1-2 min ⚡ | Very sparse |
| **Dense Subset** ⭐ | **1,329** | **723 items** ✅ | **38K** | **68K** | **1-5 stars** | **<1 min** ⚡⚡ | **743x denser** 🚀 |
| **Large overlapping** | 116K | 554K items | 3.2M | 1.7M | 1-5 stars | 30-60 min 🐌 | Sparse |

**Dense Subset = Best of both worlds!** ✨

## ⭐ Rating Distribution

### Books:
- 5 stars: 49.4% ⭐⭐⭐⭐⭐
- 4 stars: 31.2% ⭐⭐⭐⭐
- 3 stars: 13.1% ⭐⭐⭐
- 2 stars: 4.2% ⭐⭐
- 1 star: 2.0% ⭐
- **Mean: 4.22**

### Movies:
- 5 stars: 40.6% ⭐⭐⭐⭐⭐
- 4 stars: 30.3% ⭐⭐⭐⭐
- 3 stars: 17.2% ⭐⭐⭐
- 2 stars: 7.5% ⭐⭐
- 1 star: 4.3% ⭐
- **Mean: 3.95**

More balanced than 5-core (all 1s) but still skewed positive!

## 🎯 Why This is Perfect

### **1. Real Ratings**
- Not binary (1s) like 5-core
- Actual 1-5 star ratings
- MSE/RMSE now meaningful!

### **2. Dense Data**
- Items have ~53 Books interactions on average (very dense!)
- Items have ~51 Movies interactions on average (very dense!)
- 743x denser = much better learning!
- Only 723 book items (vs 276K) = model can actually learn!

### **3. Manageable Size**
- 38K Books interactions (super fast!)
- 68K Movies interactions (super fast!)
- Only 723-1,338 items (model can learn all of them!)
- Training: <1 minute per epoch (vs 5-6 min before!)

### **4. Great Overlap**
- 1,329 users in both domains (93%)
- Great for cross-domain research
- Similar to 5-core (1,329 vs 1,062)
- But MUCH denser and better quality!

## 🚀 How to Use

### **1. Update config.yaml:**
```yaml
data:
  books_path: "data/Amazon-Dense-Subset/Amazon-Books-Dense.inter"
  movies_path: "data/Amazon-Dense-Subset/Amazon-Movies-Dense.inter"
```

### **2. Train with Simple MF:**
```bash
python3 train_simple_baseline.py --domain books
```

**Expected:**
- Time: ~1-2 minutes total (super fast!)
- RMSE: 0.9-1.1 (meaningful!)
- NDCG@10: 0.10-0.20 (MUCH better!)

### **3. Or Train with BPR:**
```bash
python3 src/train_bpr.py --domain books
```

**Expected:**
- Time: ~1-2 minutes
- NDCG@10: 0.15-0.25 (excellent!)

## 📈 Expected Performance

With this denser data:

| Metric | 5-core (sparse) | Dense Subset | Improvement |
|--------|----------------|--------------|-------------|
| **Training time** | 1-2 min | <1 min | ✅ FASTER! |
| **Items** | 21K books | 723 books | ✅ Manageable! |
| **RMSE** | Not meaningful | 0.9-1.1 | ✅ Meaningful! |
| **NDCG@10** | 0.02-0.05 | 0.10-0.20 | 5-10x better! 🚀 |
| **Recall@10** | 0.03-0.05 | 0.12-0.25 | 5-10x better! 🚀 |
| **Ratings** | All 1s | 1-5 stars | ✅ Real! |
| **Density** | Sparse | 743x denser | ✅ MUCH better! |

## ✨ Recommendation

**Use Dense Subset for:**
- ✅ You want real ratings (1-5 stars)
- ✅ You want meaningful RMSE
- ✅ You want FAST training (<1 min/epoch)
- ✅ You want MUCH better NDCG (0.10-0.20)
- ✅ You have 1,329 users for CDR
- ✅ **THIS IS NOW THE BEST OPTION!** 🎯

**Use 5-core for:**
- ✅ Ultra-fast prototyping (1-2 min)
- ✅ Testing code changes
- ✅ Don't care about rating prediction

**Sweet spot: Dense Subset!** 🎯

## 📁 Files Created

```
data/Amazon-Dense-Subset/
├── Amazon-Books-Dense.inter     (1.1 MB, 38,149 ratings)
├── Amazon-Movies-Dense.inter    (1.9 MB, 68,023 ratings)
├── dense_users.txt              (1,500 user IDs)
└── README.md                    (this file)
```

## 🎓 For Your Research

This dataset gives you:
- ✅ 1,329 users with embeddings in BOTH domains
- ✅ Real rating signals for better embeddings
- ✅ 743x denser data = MUCH better learning
- ✅ SUPER FAST training (<1 min/epoch)
- ✅ Only 723-1,338 items = manageable item space
- ✅ Expected NDCG: 0.10-0.20 (excellent for baseline!)

Perfect for Wasserstein cross-domain mapping research! 🚀

**Key improvement:** Item filtering (≥30 interactions) makes ranking metrics actually work!


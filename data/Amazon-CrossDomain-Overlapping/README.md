# Amazon-CrossDomain-Overlapping Dataset

## Overview

This dataset contains Amazon product reviews for **Books** and **Movies & TV** with **complete user overlap** - every user has reviews in both domains!

## Key Statistics

### Books Domain
- **Interactions**: 3,208,863
- **Users**: 116,356
- **Items**: 553,917
- **Ratings**: 1.0 to 5.0 stars
- **Average rating**: 4.32 ⭐
- **Avg interactions per user**: 27.6

### Movies & TV Domain  
- **Interactions**: 1,716,895
- **Users**: 116,356
- **Items**: 59,762
- **Ratings**: 1.0 to 5.0 stars
- **Average rating**: 4.22 ⭐
- **Avg interactions per user**: 14.8

### Cross-Domain Perfect!
- **ALL 116,356 users appear in BOTH domains** (100% overlap!)
- This is ideal for cross-domain recommendation research
- 110x more overlapping users than the 5-core dataset

## Rating Distribution

**Books:**
- 5 stars: 59.7% ⭐⭐⭐⭐⭐
- 4 stars: 22.6% ⭐⭐⭐⭐
- 3 stars: 10.4% ⭐⭐⭐
- 2 stars: 4.2% ⭐⭐
- 1 star: 3.1% ⭐

**Movies:**
- 5 stars: 58.6% ⭐⭐⭐⭐⭐
- 4 stars: 20.2% ⭐⭐⭐⭐
- 3 stars: 10.8% ⭐⭐⭐
- 2 stars: 5.3% ⭐⭐
- 1 star: 5.1% ⭐

## Advantages

1. **Real ratings**: Explicit 1-5 star ratings (not just implicit 1s)
2. **Much larger**: 28x more Books data, 26x more Movies data
3. **Perfect overlap**: 100% of users in both domains
4. **Better for modeling**: Can use MSE loss meaningfully
5. **Better for evaluation**: Can assess both rating prediction AND ranking

## Considerations

1. **Rating bias**: Skewed toward high ratings (60% are 5 stars)
2. **Still sparse**: 99.95%+ sparsity
3. **Power users**: Top 1% account for ~20% of interactions
4. **Duplicates**: Some duplicate entries (can be cleaned)

## Comparison with 5-core Dataset

| Metric | 5-core | Overlapping | Improvement |
|--------|--------|-------------|-------------|
| Books users | 10,485 | 116,356 | 11x |
| Movies users | 5,648 | 116,356 | 21x |
| Overlapping users | 1,062 | 116,356 | **110x** |
| Books interactions | 115K | 3.2M | 28x |
| Movies interactions | 67K | 1.7M | 26x |
| Rating range | Only 1 | 1-5 stars | ✓ |

## Usage

```bash
# Train on new dataset
python src/train_single_domain.py --domain books
python src/train_single_domain.py --domain movies

# All 116,356 users will have embeddings in BOTH domains!
```

## File Format

Same format as 5-core:
```
user_id:token	item_id:token	rating:float
AVP0HXC9FG790	0001713353	5.0
A2RE7WG349NV5D	0001713353	5.0
...
```

## Perfect for Cross-Domain Research!

With 116,356 overlapping users, you can learn strong Wasserstein mappings between domains. This is **116x more data** for your cross-domain transfer learning!

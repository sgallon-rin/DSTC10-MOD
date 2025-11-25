
# Recall at position k in n candiates R_n@K 
# measure if the positive meme is ranked in the topK position of n candidates 
def recall_compute(candidates, targets, k=5): 
    if len(candidates) > k:
        candidates = candidates[:k]
    if targets not in candidates:
        return 0 
    else: 
        return 1 

# MAP: mean average precision 
def map_compute(candidates, targets, k=5): 
    if len(candidates) > k:
        candidates = candidates[:k]
    if targets not in candidates:
        return 0 
    else: 
        idx = candidates.index(targets) 
        return 1.0 / (1.0 + idx) 


if __name__ == '__main__':
    candidates = ['23', '25', '33', '65', '67']
    targets = '33' 
    print(map_compute(candidates, targets)) 
    print(recall_compute(candidates, targets)) 


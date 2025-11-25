# accuracy@k 
def accuracy_compute(candidates, targets, k):
    if len(candidates) > k:
        candidates = candidates[:k]
    if targets not in candidates:
        return 0 
    else: 
        return 1 

if __name__ == '__main__':
    print('hehe')
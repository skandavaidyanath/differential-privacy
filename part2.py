import numpy as np
from collections import defaultdict
from tqdm import tqdm


def part_a(p1=0.5, p2=0.5, size=10, seed=None):
    if seed:
        np.random.seed(seed)

    ## Create datasets
    D1 = np.random.randint(2, size=size)
    D2 = D1.copy()
    D2[0] = ~D1[0]  

    eps1 = np.log((1+p1)/2) - np.log((1-p1)/2)
    eps2 = np.log((1+p2)/2) - np.log((1-p2)/2)
    
    def algo(D, p):
        D = D.copy()
        mask = np.random.rand(size)
        mask = mask<=p  ## 1 implies we randomize
        D[mask] = np.random.randint(2, size=int(mask.sum()))
        return D.sum()

    def simulate(D, res, num_trials=100000):
        counts = defaultdict(int)
        for _ in range(num_trials):
            a = algo(D, p1)
            b = algo(D, p2)
            counts[(a, b)] += 1
        counts = {k: v/num_trials for k, v in counts.items()}
        return counts.get(res, 0)
    
    max_est = -np.inf
    for _ in tqdm(range(100)):
        res1 = algo(D1, p1)
        res2 = algo(D1, p2)
        c1 = simulate(D1, (res1, res2))
        c2 = simulate(D2, (res1, res2))

        if c1==0 or c2==0:
            continue

        eps = np.log(c1) - np.log(c2)

        if eps > max_est:
            max_est = eps
    
    print(f'Epsilon 1: {eps1}')
    print(f'Epsilon 2: {eps2}')
    print(f'Estimated Epsilon: {max_est}')
    print(f'Upper bound: {eps1 + eps2}')



if __name__ == "__main__":
    part_a()
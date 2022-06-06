import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

### EPSILON ###

def part_a():
    '''
    Identifying epsilon and plotting as a function of p
    '''
    ## The randomization probability
    p = np.array([0.1 * i for i in range(10)])

    truth = p + (1-p)/2
    lie = 1 - truth

    epsilon = np.log(truth) - np.log(lie)

    plt.plot(p, epsilon)
    plt.xlabel('Probability of randomizing')
    plt.ylabel('Epsilon')
    plt.scatter([0.5], [np.log(3)], c='red', marker='*')
    plt.text(0.5, np.log(3), f'0.5, {round(np.log(3),2)}')

    plt.show()


def part_b(p=0.5, seed=None):
    '''
    The connection between epsilon and updated beliefs 
    Inspired from https://desfontain.es/privacy/differential-privacy-in-more-detail.html
    '''
    if seed:
        np.random.seed(seed)

    p_vals = [0.1, 0.2, 0.5, 0.7, 0.9]
    c_vals = ['red', 'blue', 'green', 'purple', 'orange']
    priors = np.array([0.1 * i for i in range(11)]) ## P[D = D_in]

    for p, c in zip(p_vals, c_vals):
        eps = np.log((1+p)/2) - np.log((1-p)/2)

        upper = (np.exp(eps) * priors)/ (1 + (np.exp(eps)-1)*priors)
        lower = (priors)/ (np.exp(eps) + (1 - np.exp(eps))*priors)

        plt.plot(priors, upper, c=c)
        plt.plot(priors, lower, c=c)
    
    plt.plot(priors, priors, label='priors')
    plt.show()


def part_c(p=0.5, size=100, seed=None):
    '''
    The connection between epsilon and updated beliefs 
    Inspired from https://desfontain.es/privacy/differential-privacy-in-more-detail.html
    '''
    if seed:
        np.random.seed(seed)

    ## Create datasets
    D1 = np.random.randint(2, size=size)
    D2 = D1.copy()
    D2[0] = ~D1[0]     ### Say we only care about the first person
    eps = np.log((1+p)/2) - np.log((1-p)/2)
    priors = np.array([0.1 * i for i in range(11)]) ## P[D = D_in]

    def algo(D):
        D = D.copy()
        mask = np.random.rand(size)
        mask = mask<=p  ## 1 implies we randomize
        D[mask] = np.random.randint(2, size=int(mask.sum()))
        return D.sum()

    def simulate(D, res, num_trials=1000):
        counts = defaultdict(int)
        for _ in range(num_trials):
            x = algo(D)
            counts[x] += 1
        counts = {k:v/num_trials for k,v in counts.items()}
        return counts[res]

    x1 = algo(D1)
    x2 = algo(D2)

    v11 = simulate(D1, x1)
    v12 = simulate(D2, x1)
    p1 = (priors * v11) / (priors * v11 + (1-priors)*v12)

    v21 = simulate(D1, x2)
    v22 = simulate(D2, x2)
    p2 = (priors * v21) / (priors * v21 + (1-priors)*v22)

    upper = (np.exp(eps) * priors)/ (1 + (np.exp(eps)-1)*priors)
    lower = (priors)/ (np.exp(eps) + (1 - np.exp(eps))*priors)

    plt.plot(priors, priors, label='priors')
    plt.plot(priors, upper, label='upper')
    plt.plot(priors, lower, label='lower')

    plt.scatter(priors, p1, c='red', label='p1')
    plt.scatter(priors, p2, c='purple', label='p2')

    plt.legend()

    plt.show()


### DELTA ###
def part_d(threshold=5, num_trials=10000, eps=np.log(3), seed=None):
    '''
    The intuition behind delta
    Inspired from https://desfontain.es/privacy/almost-differential-privacy.html
    '''
    if seed:
        np.random.seed()

    def simulate():
        categories = ['tennis', 'volleyball', 'basketball', 'baseball', 'football']

        D = np.random.choice(categories[:-1], size=100, replace=True)  
        hist = Counter(D)
        hist['football'] = 1  ## Make football a rare event
        hist = {k: v + np.random.laplace(0, 1/eps) for k, v in hist.items()}
        return hist['football'] >= threshold

    failures = 0
    for _ in range(num_trials):
        if simulate():
            failures += 1
    
    print(f"Likelihood of seeing Football in D (approx delta): {round(failures/num_trials, 3)}")


if __name__ == '__main__':
    #part_a()
    #part_b()
    part_c()
    #part_d()
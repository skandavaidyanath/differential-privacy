import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

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


def part_b(p=0.5, size=100, seed=None):
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
        mask = np.random.rand(size)
        mask = mask<=p  ## 1 implies we randomize
        D[mask] = np.random.randint(2, size=int(mask.sum()))
        return D.sum()

    def simulate(D, res):
        counts = defaultdict(int)
        for _ in range(1000):
            x = algo(D)
            counts[x] += 1
        den = sum(counts.values())
        counts = {k:v/den for k,v in counts.items()}
        return counts[res]

    x1 = algo(D1.copy())
    x2 = algo(D2.copy())

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



if __name__ == '__main__':
    #part_a()
    part_b()
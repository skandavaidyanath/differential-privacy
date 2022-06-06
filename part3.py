import torch
import torch.nn as nn
import  numpy as np
from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict
import matplotlib.pyplot as plt

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def sgd(model, train_data, test_data, train_epochs=5, verbose=False):

    model.apply(init_weights)

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()

    train_dataset = torch.utils.data.TensorDataset(train_data[0], train_data[1])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32)
    
    test_dataset = torch.utils.data.TensorDataset(test_data[0], test_data[1])
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    for epoch in range(train_epochs):
        losses = []
        for i, (x, y) in enumerate(train_dataloader):
            out = model(x)
            loss = loss_fn(out, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
        
        if verbose:
            print(f'Epoch {epoch+1} | Loss: {np.mean(losses)}')
        
    avg_test_loss = 0.
    with torch.no_grad():
        for i, (x, y) in enumerate(test_dataloader):
            out = model(x)
            loss =  loss_fn(out, y)
            avg_test_loss += loss.item()
    
    avg_test_loss /= i+1
    if verbose:
        print(f"Test loss: {avg_test_loss}")
    return round(avg_test_loss, 2)


def dp_sgd(model, train_data, test_data, C=0.1, sigma=0.01, train_epochs=5, verbose=False):
    model.apply(init_weights)

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()

    train_dataset = torch.utils.data.TensorDataset(train_data[0], train_data[1])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32)
    
    test_dataset = torch.utils.data.TensorDataset(test_data[0], test_data[1])
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    for epoch in range(train_epochs):
        losses = []
        for i, (x, y) in enumerate(train_dataloader):
            out = model(x)
            loss = loss_fn(out, y)

            optimizer.zero_grad()
            loss.backward()
            
            with torch.no_grad():
                for param in model.parameters():
                    param.grad /= 32 * max(1, torch.linalg.norm(param.grad)/C)  ## (slightly) shady
                    param.grad += torch.normal(0, sigma**2 * C**2, size=param.grad.shape)
                    param.grad /= 32

            optimizer.step()
            
            losses.append(loss.item())
        
        if verbose:
            print(f'Epoch {epoch+1} | Loss: {np.mean(losses)}')
        
    avg_test_loss = 0.
    with torch.no_grad():
        for i, (x, y) in enumerate(test_dataloader):
            out = model(x)
            loss =  loss_fn(out, y)
            avg_test_loss += loss.item()
    
    avg_test_loss /= i+1

    if verbose:
        print(f"Test loss: {avg_test_loss}")
    return round(avg_test_loss, 2)
    

def find_epsilon(N, D, method='sgd', num_trials=1000, train_epochs=5):
    x1 = torch.randn(N, D)
    x2 = deepcopy(x1)
    x2[0] = torch.randn(D)
    y1 = 2*x1.sum(1, keepdim=True) + 5 + torch.randn(N, 1)
    y2 = 2*x2.sum(1, keepdim=True) + 5 + torch.randn(N, 1)

    test_x = torch.randn(N//5, D)
    test_y = 2*test_x.sum(1, keepdim=True) + 5
    
    train1 = (x1, y1)
    train2 = (x2, y2)
    test = (test_x, test_y)
    
    def algo(train_data):
        model = nn.Sequential(nn.Linear(D, 4), nn.ReLU(), nn.Linear(4, 1))
        if method == 'sgd':
            out = sgd(model, train_data, test, train_epochs)
        elif method == 'dp-sgd':
            out = dp_sgd(model, train_data, test, train_epochs=train_epochs, verbose=True)
        return out

    def simulate(train_data, res, num_trials):
        counts = defaultdict(int)
        for _ in range(num_trials):
            out = algo(train_data)
            counts[out] += 1
        counts = {k: v/num_trials for k, v in counts.items()}
        return counts.get(res, 0)

    epsilons = []
    for _ in tqdm(range(20)):
        res = algo(train1)
        if method == 'dp-sgd':
            raise
        p1 = simulate(train1, res, num_trials)
        if p1==0:
            continue
    
        p2 = simulate(train2, res, num_trials)
        if p2 == 0:
            continue
    
        eps = np.log(p1) - np.log(p2)
        epsilons.append(eps)

    #plt.hist(epsilons)
    #plt.show()

    print(f'Estimated Epsilon: {np.max(epsilons)}')

if __name__ == '__main__':
    find_epsilon(N=100, D=4, method='sgd', num_trials=2500, train_epochs=5)
    #find_epsilon(N=100, D=4, method='dp-sgd', num_trials=2500, train_epochs=50)
    
    
        
    
    
    

    

    
import numpy as np
import matplotlib.pyplot as plt

from mlp_2layer import MLP2,train

def three_blobs(n=900, seed=0):
    rng = np.random.default_rng(seed)
    means = np.array([[0,0], [2.5, 0.5], [-2.0, 1.5]])
    cov = np.array([[0.4,0.0],[0.0,0.4]])
    Xs, ys = [], []
    for k, m in enumerate(means):
        Xk = rng.multivariate_normal(m, cov, size=n//3)
        yk = np.full(n//3, k)
        Xs.append(Xk); ys.append(yk)
    return np.vstack(Xs), np.hstack(ys)

Xm, ym = three_blobs(900, seed=4)
m_multi = MLP2(in_dim=2, hidden_dim=64, out_dim=3, mode='classification', seed=3)
train(m_multi, Xm, ym, epochs=200, lr=5e-3, batch_size=64)
print("train acc:", (m_multi.forward(Xm).argmax(1)==ym).mean()) 
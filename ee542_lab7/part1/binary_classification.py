import numpy as np
import matplotlib.pyplot as plt

from mlp_2layer import MLP2,train

def rings(n=600, inner_r=0.5, gap=0.2, noise=0.06, seed=0):
    rng = np.random.default_rng(seed)
    n2 = n//2
    theta1 = rng.uniform(0, 2*np.pi, n2)
    r1 = inner_r + noise*rng.normal(size=n2)
    x1 = np.c_[r1*np.cos(theta1), r1*np.sin(theta1)]
    theta2 = rng.uniform(0, 2*np.pi, n-n2)
    r2 = inner_r + gap + noise*rng.normal(size=n-n2)
    x2 = np.c_[r2*np.cos(theta2), r2*np.sin(theta2)]
    X = np.vstack([x1, x2])
    y = np.array([0]*n2 + [1]*(n-n2))
    return X, y 

Xb, yb = rings(n=800, inner_r=0.6, gap=0.5, noise=0.07, seed=1)
m_bin = MLP2(in_dim=2, hidden_dim=64, out_dim=2, mode='classification', seed=2)
train(m_bin, Xb, yb, epochs=200, lr=5e-3, batch_size=64, weight_decay=1e-4)

probs = m_bin.forward(Xb)
preds = probs.argmax(axis=1)
acc = (preds == yb).mean()
print("train accuracy:", acc)
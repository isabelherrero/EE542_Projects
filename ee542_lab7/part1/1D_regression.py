import numpy as np
import matplotlib.pyplot as plt

from mlp_2layer import MLP2,train

rng = np.random.default_rng(0)
X = rng.uniform(-2, 2, (400, 1))
def target(x): 
    return np.maximum(0, 0.5*x + 0.2) + 0.3*np.maximum(0, -x + 0.5) 
y = target(X) + 0.05*rng.normal(size=(400,1)) 


m_reg = MLP2(1, 32, 1, mode='regression', seed=1)
losses = train(
    m_reg, X, y,
    epochs=500,
    lr=1e-2,
    batch_size=64,
    weight_decay=1e-4
)

xt = np.linspace(-2, 2, 9).reshape(-1, 1)
pred = m_reg.forward(xt)
print(np.hstack([xt, target(xt), pred])) 

#plotting
plt.scatter(X, y, s=5, alpha=0.3, label="Data")
plt.plot(xt, target(xt), label="Target")
plt.plot(xt, pred, label="Prediction")
plt.legend()
plt.savefig("results/regression.png")
plt.close()
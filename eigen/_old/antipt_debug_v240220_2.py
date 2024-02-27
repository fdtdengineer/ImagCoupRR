#%%
import numpy as np
import matplotlib.pyplot as plt


##### 線形な場合 #####

def Hamiltonian(delta=1,kappa=1, theta=0):
    iks = 1.j * kappa * np.exp(1.j * theta)
    H = np.array([[delta, iks], [iks, -delta]])
    return H

npr_delta = np.linspace(-2,2,101)
kappa = 1
theta = np.pi*0.01

npr_w = np.zeros((len(npr_delta),2), dtype=complex)
npr_v = np.zeros((len(npr_delta),2,2), dtype=complex)
npr_H = np.zeros((len(npr_delta),2,2), dtype=complex)
npr_phase = np.zeros(len(npr_delta))

for i, delta in enumerate(npr_delta):
    H = Hamiltonian(delta, kappa, theta)
    npr_H[i] = H
    w, v = np.linalg.eig(H)
    npr_w[i] = w
    npr_v[i] = v

    # 虚部の小さいほうのidxを取得
    #idx = np.argmin(np.abs(np.imag(w)))
    v_idx = v[:,0]#[:,idx]
    phase = np.angle(v_idx[1] / v_idx[0]) / np.pi
    npr_phase[i] = phase

# sort
for i in range(1, len(npr_delta)):
    d1 = npr_w[i] - npr_w[i-1]
    d2 = npr_w[i] - npr_w[i-1][::-1]
    if np.abs(d1[0]) > np.abs(d2[0]):
        npr_w[i] = npr_w[i][::-1]
        npr_v[i] = npr_v[i][:,::-1]
        npr_phase[i] = -npr_phase[i]

    
plt.figure(figsize=(5,4))
plt.plot(npr_delta, npr_w[:,0], label='w0')
plt.plot(npr_delta, npr_w[:,1], label='w1')
plt.show()

plt.figure(figsize=(5,4))
plt.plot(npr_delta, npr_phase, label='phase')
plt.show()



#%%
##### 非線形な場合 #####
from scipy.optimize import minimize

def loss(H, arg):
    omega = arg[0] + 1.j * arg[1]    
    x_0 = 1 #+ 1.j * arg[2]
    x_1 = arg[2] + 1.j * arg[3]
    x0 = np.array([x_0, x_1])

    v = H @ x0 - omega* x0
    loss = np.conj(v).T @ v

    return loss.real


H_ep = Hamiltonian(0, 1, 0)
def loss_opt(arg):
    return loss(H_ep, arg)
np.random.seed(0)
#x0 = np.random.rand(4)
x0 = np.array([0, 0, -1, 0])
res = minimize(loss_opt, x0, tol=1e-20)

omega = res.x[0] + 1.j * res.x[1]
x = np.array([1, res.x[2] + 1.j * res.x[3]])
#x = np.concatenate([np.array([1]), res.x[1:]])
x /= np.linalg.norm(x)
print("omega: ", omega)
print("x: ", x)
print("\n")

print(res)
print("\n")

# debug
w, v = np.linalg.eig(H_ep)
print("omega_theo: ", w)
print("vec_theo: \n", v)
print("\n")

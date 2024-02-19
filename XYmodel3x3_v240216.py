#%%
# 放物線のグラフを描画する
if True:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib import rc
    rc('text', usetex=False)
    fs = 18
    plt.rcParams['font.family']= 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams["font.size"] = fs # 全体のフォントサイズが変更されます。
    plt.rcParams['xtick.direction'] = 'in'#x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
    plt.rcParams['ytick.direction'] = 'in'#y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
    from mpl_toolkits.mplot3d import Axes3D
    #import ipywidgets

    filepath_output = "output\\"


eps=0.05
H0 = -np.array([
    [0,-1,-(1.+eps)],
    [-1,0,-1],
    [-(1.+eps),-1,0]
    ])

# XY model in 3x3

def Hamiltonian(x):
    out = 0
    nx = x.shape[0]
    for i in range(nx):
        for j in range(nx):
            out += H0[i,j] * np.cos(x[i]-x[j])
            #out += np.cos(x[i]-x[j])
    return out

def H(x):
    x1 = np.concatenate([np.array([0]),x])
    return Hamiltonian(x1)


nx = 3

# minimize the Hamiltonian
import scipy.optimize as opt
#x0 = np.array([-0.1,0.3])
x0 = np.array([2/3,-2/3])*np.pi
res = opt.minimize(H, x0)
#phi = (res.x -res.x[0]) / np.pi
#print(phi)
print(res)

x1 = np.concatenate([np.array([0]),res.x])
#x1 = np.array([0, 2/3,-2/3])*np.pi
#y1 = Hamiltonian(x1)
#print(y1)

def plot_quivar(npr_r, npr_phi):
    coef = 0.5
    X = np.array([0, 1, 0.5])
    Y = np.array([0, 0, np.sqrt(3)/2])
    #X, Y = np.meshgrid(X, Y)
    U = coef*npr_r*np.cos(npr_phi*np.pi - np.pi/2)
    V = coef*npr_r*np.sin(npr_phi*np.pi - np.pi/2)

    plt.figure(figsize=(4,4))
    plt.scatter(X, Y, s=npr_r**2*4000, c="turquoise", alpha=0.5)
    plt.quiver(X-U/2, Y-V/2, U, V, angles='xy', scale_units='xy', scale=1)

    for i in range(3):
        plt.text(
            X[i], Y[i]+0.35,
            "$r_{}$=".format(i+1)+str(np.round(npr_r[i], 3)),
            fontsize=14, ha='center', va='center')
        plt.text(
            X[i], Y[i]+0.45,
            "$\phi_{}$=".format(i+1)+str(np.round(npr_phi[i], 3))+"$\pi$",
            fontsize=14, ha='center', va='center')

    plt.xlim(-0.5, 2-0.5)
    plt.ylim(2-0.5, -0.5)
    #plt.gca().set_aspect('equal', adjustable='box')
    # ラベル、軸の表示を消す
    plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
    plt.tick_params(bottom=False, left=False, right=False, top=False)
    # 枠線を消す
    plt.box(False)
    plt.tight_layout()

    plt.show()

npr_r = np.ones(3)
npr_phi = (x1 - x1[0]) / np.pi
plot_quivar(npr_r, npr_phi)

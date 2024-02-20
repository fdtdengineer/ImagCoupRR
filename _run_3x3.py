
#%%
import numpy as np
import matplotlib.pyplot as plt


g = 5j

eps=0.05
H = 1.j* np.array([
    [-(2.+eps),-1,-(1.+eps)],
    [-1,-2,-1],
    [-(1.+eps),-1,-(2.+eps)]
    ]) + g*np.diag([1,1,1])

### eigenvalues and eigenvectors ###
eigenvalues, eigenvectors = np.linalg.eig(H)
print(eigenvalues.imag[np.argsort(eigenvalues.imag)][::-1])
idx0 = np.argmax(eigenvalues.imag)
eigvec0 = eigenvectors[:,idx0]
r0 = np.abs(eigvec0)
r0 = r0 / r0[0]
phi0 = np.angle(eigvec0) / np.pi
#phi0 = phi0 - phi0[0]
phi0 = (phi0 + 2) % 2 - 1

import time_evolution
#psi0 = np.zeros(H.shape[0], dtype=np.complex128)
#psi0[0] = 1+1.j
#psi0[1] = 2-1.j
#psi0[2] = -3-1.j

#seed

np.random.seed(12345678)
a0 = np.random.rand(H.shape[0]) + 1.j*np.random.rand(H.shape[0])

cme = time_evolution.CoupledModeEquation(H, dt=0.01, tmax=500)
cme.set_initial_state(a0)
cme.solve_stuartlandau(beta=1e-3)

cme.plot("psiAbs")
cme.plot("psiAbsRel")
cme.plot("psiPhaseRel")
#cme.plot("psiPhase")    

ave_r = cme.get_average(key="psiReal", num_data=1)
ave_rRel = cme.get_average(key="psiAbsRel", num_data=1)
ave_phi = cme.get_average(key="psiPhaseRel", num_data=1)

#np.round(ave_rRel, 3), np.round(ave_phi, 3)


# quiver
# ave_rRel, ave_phi

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

plot_quivar(r0, phi0)
plot_quivar(ave_rRel, ave_phi)


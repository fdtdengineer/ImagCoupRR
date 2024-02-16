
#%%
import numpy as np


g = 3j
H = 1.j* np.array([
    [-2,-1,-1],
    [-1,-2,-1],
    [-1,-1,-2]
    ]) + g*np.diag([1,1,1])

### eigenvalues and eigenvectors ###
eigenvalues, eigenvectors = np.linalg.eig(H)
print(eigenvalues.imag[np.argsort(eigenvalues.imag)][::-1])


import time_evolution
psi0 = np.zeros(H.shape[0], dtype=np.complex128)
psi0[0] = 1+1.j
psi0[1] = 2-1.j
psi0[2] = -3-1.j


cme = time_evolution.CoupledModeEquation(H, dt=0.01, tmax=10)
cme.set_initial_state(psi0)
cme.solve_stuartlandau(beta=1e-2)

cme.plot("psiAbs")
cme.plot("psiAbsRel")
cme.plot("psiPhaseRel")
#cme.plot("psiPhase")    

ave_r = cme.get_average(key="psiReal", num_data=30)
ave_phi = cme.get_average(key="psiPhaseRel", num_data=30)



# %%

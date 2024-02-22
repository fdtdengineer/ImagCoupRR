
#%%
import numpy as np

# rand
np.random.seed(0)

gain=0 #10
n1 = 3 # サイズ横
n2 = 3 # サイズ縦
npr_elem_gh = np.random.randn(n2,n1-1) # 横結合
npr_elem_gv = np.random.randn(n1*(n2-1)) # 縦結合


# 非対角成分が npr_g1 である n1 x n1 行列 を n2 個作る
npr_gh_block = np.zeros((n2,n1,n1), dtype=complex)
for i in range(n2):
    npr_gh_block[i] = np.diagflat(npr_elem_gh[i], k=1) + np.diagflat(npr_elem_gh[i], k=-1)

# npr_gh の各要素をブロック対角行列として結合
npr_gh = np.zeros((n2*n1,n2*n1), dtype=complex)
for i in range(n2):
    npr_gh[i*n1:(i+1)*n1, i*n1:(i+1)*n1] = npr_gh_block[i]

npr_gv = np.zeros((n1*n2,n1*n2), dtype=complex)
for i in range(n1*(n2-1)):
    npr_gv[i,i+n1] = npr_elem_gv[i]
    npr_gv[i+n1,i] = npr_elem_gv[i]

# np.round(npr_g,2).real # debug
npr_g = npr_gh + npr_gv

# npr_g の対角成分を、それぞれの行の非対角成分の絶対値の和にする
npr_g_abs = np.abs(npr_g)
npr_elem_gdiag = np.sum(npr_g_abs, axis=1)
npr_gdiag = np.diagflat(npr_elem_gdiag)

H = 1.j*npr_g -1.j*npr_gdiag
Hinit = H.copy()

Hgain = 1.j*np.ones(n1*n2)*gain
H = H + np.diag(Hgain)

### eigenvalues and eigenvectors ###
eigenvalues, eigenvectors = np.linalg.eig(H)
#print(np.round(eigenvalues.imag, 3))
print(eigenvalues.imag[np.argsort(eigenvalues.imag)][::-1][:2])



import time_evolution
#psi0 = np.zeros(n1*n2, dtype=np.complex128)
#psi0[0] = 1.
psi0 = np.random.randn(n1*n2) + 1.j*np.random.randn(n1*n2)

cme = time_evolution.CoupledModeEquation(H, dt=0.01, tmax=200)
cme.set_initial_state(psi0)
cme.solve_stuartlandau(beta=1e-3)

cme.plot("psiAbs")
cme.plot("psiAbsRel")
cme.plot("psiPhaseRel")
r0 = cme.get_average(key="psiAbsRel", num_data=1)
phi0 = cme.get_average(key="psiPhaseRel", num_data=1)









### eigenvalues and eigenvectors ###
eigenvalues, eigenvectors = np.linalg.eig(H)
print(np.round(eigenvalues.imag, 3))
#print("max imag:"+str(np.max(eigenvalues.imag)))

# get the index which has the highest imag part of eigenvalue
idx = np.argmax(np.imag(eigenvalues))
eig0 = eigenvalues[idx]
vec0 = eigenvectors[:,idx]
#print(eig0)

# plot the eigenvector
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# phase
phase = np.angle(vec0) / np.pi
phase = phase - phase[0]
phase = np.array([x-2 if x>= 1.5 else x for x in phase])
phase = np.array([x+2 if x<=-0.5 else x for x in phase])
phase = phase.reshape(-1,n1)

# mod 2
#phase = np.mod(phase, 2)

# heatmap
"""
fig, ax = plt.subplots()
cax = ax.matshow(phase, cmap=cm.hot, vmin=-1, vmax=1)
fig.colorbar(cax)
plt.show()


plt.figure()
plt.bar(range(len(vec0)), np.abs(vec0))
plt.show()
"""

######################

phi = np.array(phase, dtype=complex)*np.pi
"""
phi_i = phi.reshape(-1,1)
phi_j = phi.reshape(1,-1)
phi_ij = phi_i - phi_j

vtensor = np.cos(phi_ij)
# round
vtensor = np.round(vtensor, 2).real
#print("vtensor:\n", vtensor)

loss = H * vtensor
loss = np.sum(loss)
print("loss:", loss)
"""




phi_i = phi0.reshape(-1,1) #* np.pi
phi_j = phi0.reshape(1,-1) #* np.pi
phi_ij = phi_i - phi_j

vtensor = np.cos(phi_ij)

Hloss = Hinit.imag
# Hloss の対角成分を 0 にする
#np.fill_diagonal(Hloss, 0)
loss = Hloss * vtensor
loss = -np.sum(loss)/2
print("loss:", loss)



# quiver H
# coef = 0.5
coef = np.abs(vec0).reshape(n1,n2)
coef = coef / np.max(coef) * 0.5
X = np.arange(0, n1, 1)
Y = np.arange(0, n2, 1)
X, Y = np.meshgrid(X, Y)
U = coef*np.cos(phi.real - np.pi/2)
V = coef*np.sin(phi.real - np.pi/2)

plt.figure(figsize=(4,4))
plt.quiver(X-U/2, Y-V/2, U, V, angles='xy', scale_units='xy', scale=1)
plt.xlim(-0.5, n1-0.5)
plt.ylim(n2-0.5, -0.5)
#plt.xlabel('location x')
#plt.ylabel('location y')
plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
plt.tick_params(bottom=False, left=False, right=False, top=False)
#plt.box(False)
plt.tight_layout()

plt.show()


# quiver TimeEvolution
coef = r0.reshape(n1,n2)
coef = coef / np.max(coef) * 0.5
X = np.arange(0, n1, 1)
Y = np.arange(0, n2, 1)
X, Y = np.meshgrid(X, Y)
phi0 = (phi0*np.pi).reshape(n1,n2)
U = coef*np.cos(phi0 - np.pi/2)
V = coef*np.sin(phi0 - np.pi/2)

plt.figure(figsize=(4,4))
plt.quiver(X-U/2, Y-V/2, U, V, angles='xy', scale_units='xy', scale=1)
plt.xlim(-0.5, n1-0.5)
plt.ylim(n2-0.5, -0.5)
#plt.xlabel('location x')
#plt.ylabel('location y')
plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
plt.tick_params(bottom=False, left=False, right=False, top=False)
#plt.box(False)
plt.tight_layout()


plt.show()


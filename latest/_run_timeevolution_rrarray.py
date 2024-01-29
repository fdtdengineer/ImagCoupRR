
#%%
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

    filepath_output = "fig\\"

    import rrarray
    import time_evolution


if __name__ == "__main__":
    omega=0
    #npr_Delta = np.array([-2,2,-1,0,1])*1
    npr_Delta = np.array([-2,-1,0,1,2])
    #npr_Delta = np.array([-5,-1,0,1,5])
    #npr_Delta = np.array([-1,-1,-2,-1,0,1,2,1,1])
    #npr_Delta = np.zeros(5)
    #npr_Delta = np.zeros(10)
    
    #np.random.seed(seed=12345678)
    #npr_Delta = np.random.randn(15)*0.1
    n = npr_Delta.shape[0]

    #npr_eta = np.array([0, 0., 0., 0., 0])*0.5
    npr_eta = np.zeros(n)
    kappa = 1#3.5 #0.35#10*(1j) # dummy because of the sweep
    theta = 1*np.pi#/2*0.8 #0.1*np.pi
    kappa2 = 0#kappa #*(-3) #0.5*(-1j) #無効
    theta2 = 0 # 0.1*np.pi
    boundary = "periodic"

    cls_rr = rrarray.RRarray(n, omega, npr_Delta, npr_eta, kappa, theta, kappa2, theta2, savefig=True, boundary=boundary)
    H = cls_rr.get_Hamiltonian()

    # for debug
    eigval, eigvec = cls_rr.solve()
    k_min = eigval.imag.argmax()
    eigval0 = eigval[k_min]
    eigvec0 = eigvec[:, k_min]
    e_i_phi = eigvec0 / np.array([abs(x) for x in eigvec0])
    Rinf = np.abs(np.sum(e_i_phi)) / n
    Vinf = np.sum(np.log(n*np.abs(eigvec0**2) )**2) / n**2
    print("Rinf = ", Rinf, ", Vinf = ", Vinf)
    
    #psi0 = np.array([1, 1, 1], dtype=np.complex)
    #psi0 = np.array([1, 0, 0, 0, 0], dtype=complex)
    #psi0 = np.zeros(n, dtype=complex)
    #psi0[0] = 1
    np.random.seed(seed=12345678+1)
    psir = np.random.randn(n)
    np.random.seed(seed=12345678+2)
    psii = np.random.randn(n)
    psi0 = psir + 1j*psii
    cme = time_evolution.CoupledModeEquation(H, dt=0.01, tmax=20)
    
    cme.set_initial_state(psi0)
    cme.solve()
    cme.plot("psiAbs")
    cme.plot("psiAbsRel")#, list_ylim=[0, 2])
    #cme.plot("psiPhase")
    cme.plot("psiPhaseRel")

    print("finished")

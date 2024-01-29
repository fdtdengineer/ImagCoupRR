
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
    npr_Delta = np.array([-2,2,-1,0,1])*0.2
    #npr_Delta = np.array([-2,-1,0,1,2])
    #npr_Delta = np.array([-5,-1,0,1,5])
    #npr_Delta = np.array([-1,-1,-2,-1,0,1,2,1,1])
    #npr_Delta = np.zeros(5)
    #npr_Delta = np.zeros(10)
    n = npr_Delta.shape[0]

    #npr_eta = np.array([0, 0., 0., 0., 0])*0.5
    npr_eta = np.zeros(n)
    kappa = 1#3.5 #0.35#10*(1j) # dummy because of the sweep
    theta = np.pi*5/6 #0.1*np.pi
    kappa2 = 0#kappa #*(-3) #0.5*(-1j) #無効
    theta2 = 0 # 0.1*np.pi

    cls_rr = rrarray.RRarray(n, omega, npr_Delta, npr_eta, kappa, theta, kappa2, theta2, savefig=True)
    H = cls_rr.get_Hamiltonian()
    #H = np.conjugate(H.T)
    
    #psi0 = np.array([1, 1, 1], dtype=np.complex)
    psi0 = np.array([1, 0, 0, 0, 0], dtype=np.complex)
    #psi0 = np.zeros(20, dtype=np.complex)
    #psi0[0] = 1
    #psi0 = np.random.randn(n) + 1j*np.random.randn(n)

    cme = time_evolution.CoupledModeEquation(H, dt=0.01, tmax=20)
    
    cme.set_initial_state(psi0)
    cme.solve()
    cme.plot("psiAbsRel")
    #cme.plot("psiAbsRel")
    #cme.plot("psiPhase")
    #cme.plot("psiPhaseRel")

    print("finished")

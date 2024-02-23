#%%
if True:
    import rrarray
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
    filepath_output = "output\\"

    filepath_output = "fig\\"
    import rrarray


if __name__ == "__main__":
    gain = 5#398
    delta = 0.0
    #npr_Delta = np.array([-2,2,-1,0,1])
    #npr_Delta = np.array([-2,-1,0,1,2])
    #npr_Delta = np.array([-1,0.5,1])
    npr_Delta = np.array([1,2,4])#+10
    
    
    #npr_Delta = np.zeros(5)
    n = npr_Delta.shape[0]
    #npr_eta = -np.array([0., 0., 0., 0., 0.])*0.5
    npr_eta = np.ones(n)*gain
    kappa = 5 #
    theta = np.pi#0.2*np.pi #
    kappa2 = 0 #
    theta2 = 0 #

    #npr_kappa = np.linspace(0, 2, 201)
    npr_kappa = np.linspace(0, 5, 101)
    
    npr_sweep = np.array([npr_kappa]) # kappa と kappa2 を同時に変化させる

    #cls_rr = rrarray.RRarray(n, delta, npr_Delta, npr_eta, kappa, theta, kappa2, theta2, savefig=True, boundary="open")
    #cls_rr.sweep(npr_sweep, list_keys=["kappa"])
    #cls_rr.plot_eigval_sweep()

    import time_evolution
    #seed
    np.random.seed(12345678)

    # initial state
    a0 = np.random.rand(n) + 1.j*np.random.rand(n)

    cls_rr2 = rrarray.RRarray(n, delta, npr_Delta, npr_eta, kappa, theta, kappa2, theta2, savefig=True, boundary="open")
    cme = time_evolution.CoupledModeEquation(cls_rr2.H, dt=0.01, tmax=200)
    cme.set_initial_state(a0)
    cme.solve_stuartlandau(beta=1e-3)
    #cme.plot("psiReal")
    #cme.plot("psiAbsRel")
    #cme.plot("psiPhaseRel")
    #cme.plot("psiPhase")    

    ave_r = cme.get_average(key="psiReal", num_data=1)
    ave_rRel = cme.get_average(key="psiAbsRel", num_data=1)
    ave_phi = cme.get_average(key="psiPhaseRel", num_data=1)

    #save
    cme.save_csv(filename="cme.csv")
    
    dict_fft = cme.get_fft(key="psiReal", num_data=10000)
    df_fft = dict_fft["df"]
    peak = dict_fft["peak"]

    plt.plot(df_fft)
    plt.xlim(0, 5)

    print(peak)
    #print("end")

    #%%
    E = peak[0] # real part of frequency
    phi = ave_rRel*np.exp(1.j*ave_phi*np.pi)
    phi = phi / np.linalg.norm(phi)
    phi = phi * np.exp(-1.j*np.angle(phi[0]))

    v1 = E * phi
    v2 = cls_rr2.H @ phi

    print(v1)
    print(v2)
    print("\n")

    # 解析解（線形）
    eigval, eigvec = np.linalg.eig(cls_rr2.H)
    idx_minloss = eigval.imag.argmax()
    #print("eigval, eigvec")
    eval = eigval[idx_minloss]
    evec = eigvec[:, idx_minloss]
    print(eval * evec)
    print(cls_rr2.H @ evec)
    print("\n")

    print(eval * phi)

    #print(eigval[idx_minloss])
    #print(eigvec[:, idx_minloss])

    #%%






# %%

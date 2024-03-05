#%%
if True:
    import rrarray
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib import rc
    rc('text', usetex=False)
    fs = 16
    plt.rcParams['font.family']= 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams["font.size"] = fs # 全体のフォントサイズが変更されます。
    plt.rcParams['xtick.direction'] = 'in'#x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
    plt.rcParams['ytick.direction'] = 'in'#y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
    filepath_output = "output\\"

    filepath_output = "fig\\"
    import rrarray


if __name__ == "__main__":
    gain = 3 #2.5#1.2#0.#398
    gs = 1e-3

    delta = 0.0
    npr_Delta = np.array([1,2,4])#+10
    
    
    #npr_Delta = np.zeros(5)
    n = npr_Delta.shape[0]
    #npr_eta = -np.array([0., 0., 0., 0., 0.])*0.5
    npr_eta = np.ones(n)*gain
    kappa = 1.006 # 0.28 #0.9#1.5
    theta = np.pi#0.2*np.pi #
    kappa2 = 0 #
    theta2 = 0 #

    npr_kappa = np.linspace(0, 2, 201)
    #npr_kappa = np.linspace(0, 5, 101)
    
    npr_sweep = np.array([npr_kappa]) # kappa と kappa2 を同時に変化させる

    cls_rr = rrarray.RRarray(n, delta, npr_Delta, npr_eta, kappa, theta, kappa2, theta2, savefig=True, boundary="open")
    cls_rr.sweep(npr_sweep, list_keys=["kappa"])
    cls_rr.plot_eigval_sweep()

    import time_evolution
    #seed
    np.random.seed(12345678)

    # initial state
    a0 = np.random.rand(n) + 1.j*np.random.rand(n)

    cls_rr2 = rrarray.RRarray(n, delta, npr_Delta, npr_eta, kappa, theta, kappa2, theta2, savefig=True, boundary="open")
    cme = time_evolution.CoupledModeEquation(cls_rr2.H, dt=0.01, tmax=20)
    
    cme.set_initial_state(a0)
    cme.solve_stuartlandau(beta=gs)
    cme.plot("psiAbs",yscale="log")
    cme.plot("psiAbsRel")
    cme.plot("psiRealRel")
    #cme.plot("psiPhaseRel")
    #cme.plot("psiPhase")    

    ave_r = cme.get_average(key="psiReal", num_data=1)
    ave_rAbs = cme.get_average(key="psiAbs", num_data=1)
    ave_rRel = cme.get_average(key="psiAbsRel", num_data=1)
    ave_phi = cme.get_average(key="psiPhaseRel", num_data=1)

    #save
    cme.save_csv(filename="cme.csv")
    
    dict_fft = cme.get_fft(num_data=250, type_peak="all")
    df_fft = dict_fft["df"]
    peak = dict_fft["peak"]
    decay = dict_fft["decay"]

    plt.figure(figsize=(4, 3))
    plt.plot(df_fft)
    plt.xlim(0, 5)
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    

    print(peak)
    #print("end")

    #%%
    idx_sol=0
    E = peak[idx_sol] + 1.j*decay[idx_sol] # real part of frequency
    phi = ave_rAbs*np.exp(1.j*ave_phi*np.pi)
    phi = phi * np.exp(-1.j*np.angle(phi[0]))
    phi0 = phi / np.linalg.norm(phi)
    v1 = E * phi
    v2 = cls_rr2.H @ phi - 1.j*gs * np.abs(phi)**2 * phi


    print("Time-simulated solution")
    print(v1)
    print(v2)
    print("Error ratio: " + str(np.round(np.linalg.norm(v1-v2)/np.linalg.norm(v2), 5)))
    print("\n")
    print("Frequency: " + str(E))
    print("\n")

    # 解析解（線形）
    eigval, eigvec = np.linalg.eig(cls_rr2.H)
    idx_minloss = eigval.imag.argmax()
    #print("eigval, eigvec")
    eval = eigval[idx_minloss]
    evec = eigvec[:, idx_minloss]
    #evec /= evec[0]
    print("Analytical solution (linear)")
    print(eval * evec)
    print(cls_rr2.H @ evec)
    print("\n")

    print(eval * phi0)
    print("\n")
    print("Error between Tie and Analytical: ")
    print(phi0 / evec)
    amp_theo = np.sqrt(np.abs(eval.imag) / gs)
    #%%
    # phi0 は各サイトの振幅が揃う傾向にあるらしい
    v_t = (3 * np.conj(phi0) * phi0).real
    v_a = (3 * np.conj(evec) * evec).real
    sigma_t = np.std(np.log(v_t))
    sigma_a = np.std(np.log(v_a))
    print(sigma_t, sigma_a)


    #%%
    a = np.array([1, 1, 1])
    E = 1
    amp_theo = 100#np.sqrt(np.abs(eval.imag) / gs)
    cls_rr_3 = rrarray.RRarray(n, delta, npr_Delta, npr_eta, kappa, theta, kappa2, theta2, savefig=True, boundary="open")
    H = cls_rr_3.H
    v = H@a - E*a
    loss = (np.conj(v).T @ v).real + np.linalg.norm(np.abs(a) - amp_theo)**2
    






# %%

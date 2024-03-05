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

    filepath_output = "out\\"
    import rrarray


if __name__ == "__main__":
    gain = 5
    gs = 1e-3

    delta = 0.0
    npr_Delta = np.array([1,2,4])#+10
       
    n = npr_Delta.shape[0]
    #kappa = 1 #
    theta = np.pi#0.2*np.pi #
    kappa2 = 0 #
    theta2 = 0 #
    npr_gain = np.linspace(0, 2, 3)
    npr_kappa = np.linspace(0, 2, 21)
    
    import time_evolution
    #seed
    np.random.seed(12345678)

    # run
    a0 = np.random.rand(n) + 1.j*np.random.rand(n)

    arr_Er = np.zeros((npr_kappa.shape[0], npr_gain.shape[0], n), dtype=float)
    arr_Ei = np.zeros((npr_kappa.shape[0], npr_gain.shape[0], n), dtype=float)

    for i, gain in enumerate(npr_gain):
        npr_eta = np.ones(n)*gain
        for j, kappa in enumerate(npr_kappa):
            cls_rr2 = rrarray.RRarray(n, delta, npr_Delta, npr_eta, kappa, theta, kappa2, theta2, savefig=True, boundary="open")
            eigval, _ = cls_rr2.solve()
            

            cme = time_evolution.CoupledModeEquation(cls_rr2.H, dt=0.01, tmax=500)
            cme.set_initial_state(a0)
            cme.solve_stuartlandau(beta=gs)
            #cme.solve_3rdwnorm(beta=gs)          

            ave_r = cme.get_average(key="psiReal", num_data=1)
            ave_rAbs = cme.get_average(key="psiAbs", num_data=1)
            ave_rRel = cme.get_average(key="psiAbsRel", num_data=1)
            ave_phi = cme.get_average(key="psiPhaseRel", num_data=1)
        
            dict_fft = cme.get_fft(num_data=25000, type_peak="all")
            df_fft = dict_fft["df"]
            peak = dict_fft["peak"]
            decay = dict_fft["decay"]

            #peak = np.array([peak.tolist() + eigval.real[eigval.imag < 0].tolist()])

            arr_Er[j,i] = peak#[idx_sol]
            arr_Ei[j,i] = decay#[idx_sol]
            #E = peak[idx_sol] + 1.j*decay[idx_sol]
            #phi = ave_rAbs*np.exp(1.j*ave_phi*np.pi)
            #phi = phi * np.exp(-1.j*np.angle(phi[0]))
            #phi0 = phi / np.linalg.norm(phi)
            #v1 = E * phi
            #v2 = cls_rr2.H @ phi - 1.j*gs * np.abs(phi)**2 * phi

    columns = [f"gain_{str(np.round(gain,2))}" for gain in npr_gain]
    #df_re = pd.DataFrame(arr_Er, index=npr_kappa, columns=columns)
    #df_im = pd.DataFrame(arr_Ei, index=npr_kappa, columns=columns)

    #save
    #df_re.to_csv(filepath_output + "df_re.csv")
    #df_im.to_csv(filepath_output + "df_im.csv")

    print("done")




# %%

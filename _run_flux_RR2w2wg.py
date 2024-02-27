#%%
if True:
    import rrarray
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib import rc
    rc('text', usetex=False)
    fs = 15
    plt.rcParams['font.family']= 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams["font.size"] = fs # 全体のフォントサイズが変更されます。
    plt.rcParams['xtick.direction'] = 'in'#x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
    plt.rcParams['ytick.direction'] = 'in'#y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
    filepath_output = "output\\"


if __name__ == "__main__":
    omega=0
    #npr_Delta = -np.array([-2,2,-1,0,1])
    npr_Delta = -np.array([-1,1])
    #npr_Delta = np.array([-5,-1,0,1,5])

    #npr_Delta = np.zeros(5)
    n = npr_Delta.shape[0]
    #npr_eta = -np.array([1, 0, 0, 0, 1])
    npr_eta = -np.array([1, 1])*0.1
    npr_kappa = np.array([1,1])*1#0.5
    npr_theta = np.array([1,1])*np.pi*0
    npr_t = np.array([1,1])*1

    # sweep kappa
    arr_kappa = np.linspace(0, 1, 101)
    arr_sweep = np.outer(arr_kappa, npr_kappa) # np.array([arr_kappa])
    arr_sweep = arr_sweep.reshape(1,-1, 2)
    cls_rr2wg_sweep = rrarray.RR2w2wg(
        n, omega, npr_Delta, npr_eta, npr_kappa, npr_theta, npr_t, savefig=True
        )
    cls_rr2wg_sweep.sweep(arr_sweep, ["npr_kappa"])
    cls_rr2wg_sweep.plot_eigval_sweep() #list_ylim=[-4, 1])
    
    # flux
    npr_kappa = np.array([1,1])*0.5
    npr_omega = np.linspace(-2,2,201)
    cls_rr2wg = rrarray.RR2w2wg(
        n, omega, npr_Delta, npr_eta, npr_kappa, npr_theta, npr_t, savefig=True
        )
    cls_rr2wg.solve_flux(npr_omega)
    cls_rr2wg.plot_flux()

    xmax = 2 #0.5
    xmin = -xmax
    p0 = [1, 0, 1, 3]
    cls_rr2wg.plot_flux(plottype="both",ylim=[0,1])
    cls_rr2wg.fit_by_lorentzian(plottype="T",xmin=xmin,xmax=xmax,p0=p0)

    
    print("end")






# %%

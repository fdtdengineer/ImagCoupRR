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
    omega = 0
    npr_Delta = np.array([1,2,4])
    #npr_Delta = np.zeros(5)
    n = npr_Delta.shape[0]
    npr_eta = np.array([1., 1., 1])*0.
    kappa = 1#
    theta = 0.0*np.pi #np.pi #0.2*np.pi #
    kappa2 = 0 #
    theta2 = 0 #

    npr_sweep_kappa = np.linspace(0, 2, 51)
    arr_sweep = np.array([npr_sweep_kappa])
    
    cls_rr = rrarray.RRarray(n, omega, npr_Delta, npr_eta, kappa, theta, kappa2, theta2, savefig=True)
    cls_rr.sweep(arr_sweep, list_keys=["kappa"])

    cls_rr.plot_eigval_sweep(npr_x=npr_sweep_kappa, xlabel="$\kappa$",list_ylim=[-4,1])
    cls_rr.show_eigvec(npr_sweep_kappa.shape[0] - 1)
    




    print("end")






# %%

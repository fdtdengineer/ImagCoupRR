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
    omega = 0
    npr_Delta = np.array([-2,2,-1,0,1])
    #npr_Delta = np.array([-2,-1,0,1,2])
    #npr_Delta = np.array([-2,-1,0,1,2])
    #npr_Delta = np.array([-5,0,0,0,5])
    #npr_Delta = np.array([-5,-1,0,1,5])
    #npr_Delta = np.zeros(5)
    n = npr_Delta.shape[0]
    npr_eta = -np.array([0., 0., 0., 0., 0.])*0.5
    kappa = 1 #
    theta = np.pi#0.2*np.pi #
    kappa2 = 0 #
    theta2 = 0 #

    npr_sweep_delta = np.linspace(0, 2, 51)
    arr_delta = np.outer(npr_sweep_delta, npr_Delta)
    arr_eta = np.outer(npr_sweep_delta, npr_eta)
    arr_sweep = np.array([arr_delta, arr_eta])
    
    cls_rr = rrarray.RRarray(n, omega, npr_Delta, npr_eta, kappa, theta, kappa2, theta2, savefig=True)
    cls_rr.sweep(arr_sweep, list_keys=["npr_delta","npr_eta"])

    cls_rr.plot_eigval_sweep(npr_x=npr_sweep_delta, xlabel="$\delta$",list_ylim=[-8,1])
    cls_rr.show_eigvec(npr_sweep_delta.shape[0] - 1)
    




    print("end")






# %%

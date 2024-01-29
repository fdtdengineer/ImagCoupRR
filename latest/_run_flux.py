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


if __name__ == "__main__":
    omega=0
    #npr_Delta = -np.array([-2,2,-1,0,1])
    npr_Delta = -np.array([-2,-1,0,1,2])
    #npr_Delta = np.array([-5,-1,0,1,5])

    #npr_Delta = np.zeros(5)
    n = npr_Delta.shape[0]

    npr_eta = np.array([1, 0., 0., 0., 1])*0.5
    kappa = 3.5#0.35#10*(1j) # dummy because of the sweep
    theta = 0 # 0.1*np.pi
    kappa2 = 0#kappa #*(-3) #0.5*(-1j) #無効
    theta2 = 0 # 0.1*np.pi
    npr_omega = np.linspace(-5,5,201)

    cls_rr = rrarray.RRarray(n, omega, npr_Delta, npr_eta, kappa, theta, kappa2, theta2, savefig=True)
    cls_rr.solve_flux(npr_omega)
    cls_rr.plot_flux()

    xmax = 0.5
    xmin = -xmax
    p0 = [1, 0, 1, 3]
    cls_rr.fit_by_lorentzian(plottype="T",xmin=xmin,xmax=xmax,p0=p0)
    
    print("end")






# %%

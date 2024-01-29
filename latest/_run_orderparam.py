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
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from mpl_toolkits.mplot3d import Axes3D

    filepath_output = "fig\\"
    import rrarray


if __name__ == "__main__":
    delta=0
    #npr_Delta = np.array([-1,-5,3,-1,5])

    #npr_Delta = np.array([-2,-1,0,1,2+1e-3])
    npr_Delta = np.array([-2,-1,0,1,2])
    #npr_Delta = np.array([-2,2,-1,0,1])
    #npr_Delta = np.array([-1,-1,-2,-1,0,1,2,1,1])
    
    #np.random.seed(seed=12345678)
    #npr_Delta = np.random.randn(15)
    
    #npr_Delta = np.zeros(5)
    #npr_Delta = np.zeros(10)
    n = npr_Delta.shape[0]

    #npr_eta = np.array([0, 0., 0., 0., 0])*0.5
    npr_eta = np.zeros(n)
    kappa = 1#3.5 #0.35#10*(1j) # dummy because of the sweep
    theta = np.pi #0.1*np.pi
    kappa2 = 0#kappa #*(-3) #0.5*(-1j) #無効
    theta2 = 0 # 0.1*np.pi
    boundary = "periodic"

    npr_sweep_delta = np.linspace(0, 1, 51)
    npr_sweep_theta = np.linspace(0, 2, 41)*np.pi
    #npr_sweep_delta = np.linspace(1, 1, 1)
    #npr_sweep_theta = np.linspace(1, 1, 1)*np.pi*5/6
    
    npr_Rinf = np.zeros((npr_sweep_delta.shape[0], npr_sweep_theta.shape[0]))
    npr_Vinf = np.zeros((npr_sweep_delta.shape[0], npr_sweep_theta.shape[0]))

    for i, delta in enumerate(npr_sweep_delta):
        for j, theta in enumerate(npr_sweep_theta):
            cls_rr = rrarray.RRarray(n, 0, delta*npr_Delta, delta*npr_eta, kappa, theta, kappa2, theta2, savefig=True, boundary=boundary)
            eigval, eigvec = cls_rr.solve()
            k_min = eigval.imag.argmax()

            eigval0 = eigval[k_min]
            eigvec0 = eigvec[:, k_min]

            # check if there is degeneracy of imaginary part of eigval0
            eigval_check = eigval - eigval0
            if np.sum(((eigval_check.imag)**2 < 1e-10)*1) > 1:
                npr_Rinf[i, j] = np.nan
                npr_Vinf[i, j] = np.nan
            else:
                e_i_phi = eigvec0 / np.array([abs(x) for x in eigvec0])
                Rinf = np.abs(np.sum(e_i_phi)) / n
                npr_Rinf[i, j] = Rinf

                #Vinf = np.sum(np.log(np.sqrt(n)*np.abs(eigvec0))**2) / n
                Vinf = np.sum(np.log(n*np.abs(eigvec0**2) )**2) / n**2

                npr_Vinf[i, j] = Vinf
    df_Rinf = pd.DataFrame(npr_Rinf, index=npr_sweep_delta, columns=npr_sweep_theta/np.pi)
    df_Vinf = pd.DataFrame(npr_Vinf, index=npr_sweep_delta, columns=npr_sweep_theta/np.pi)


    # plot as a heatmap
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)
    cax = ax.imshow(df_Rinf, interpolation='nearest', cmap=cm.viridis, vmin=0, vmax=1)
    divider = make_axes_locatable(ax)    
    cbar = fig.colorbar(cax, ticks=[0,1], shrink=0.6)    
    ax.set_xticks(
        np.linspace(0, npr_sweep_theta.shape[0]-1, 5), 
        ['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$']
    )
    ax.set_yticks(
        np.linspace(0, npr_sweep_delta.shape[0]-1, 5),
        np.linspace(0, npr_sweep_delta[-1], 5),
    )
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$\delta$")
    cbar.set_label(r"$R_\infty$")
    fig.tight_layout()
    fig.savefig(filepath_output + "Rinf.svg", transparent=True)
    plt.show()
    plt.close()

    # plot as a heatmap
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)
    cax = ax.imshow(df_Vinf, interpolation='nearest', cmap=cm.viridis_r, vmin=0, vmax=0.1) #np.max(df_Vinf.values)
    divider = make_axes_locatable(ax)
    cbar = fig.colorbar(cax, ticks=[0,0.1   ], shrink=0.6)
    ax.set_xticks(
        np.linspace(0, npr_sweep_theta.shape[0]-1, 5), 
        ['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$']
    )
    ax.set_yticks(
        np.linspace(0, npr_sweep_delta.shape[0]-1, 5),
        np.linspace(0, npr_sweep_delta[-1], 5),
    )
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$\delta$")
    cbar.set_label(r"$V_\infty$")
    fig.tight_layout()
    fig.savefig(filepath_output + "Vinf.svg", transparent=True)
    plt.show()
    plt.close()

    # closssection
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)
    plt.plot(df_Rinf[1.00], color="black")
    ax.set_xlabel(r"$\delta$")
    ax.set_ylabel(r"$R_\infty$")
    fig.tight_layout()
    fig.savefig(filepath_output + "Rinf_cross_pi.svg", transparent=True)
    plt.show()

    print("Done")

# %%

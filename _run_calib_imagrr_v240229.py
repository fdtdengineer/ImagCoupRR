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
    from scipy.optimize import minimize


if __name__ == "__main__":
    gain = 2.5 #398
    gs = 1e-3

    delta = 0.0
    npr_Delta = np.array([1,2,4])#+10
    
    n = npr_Delta.shape[0]
    npr_eta = np.ones(n)*gain
    #kappa = 0.5 #
    theta = np.pi#0.2*np.pi #
    kappa2 = 0 #
    theta2 = 0 #

    npr_kappa = np.linspace(0, 2, 21)
    #npr_sweep = np.array([npr_kappa]) # kappa と kappa2 を同時に変化させる

    def loss(H, arg, beta=gs):
        omega = arg[0] #+ 1.j * arg[1]
        x_0 = arg[2] #+ 1.j * arg[2]
        x_1 = arg[3] + 1.j * arg[4]
        x_2 = arg[5] + 1.j * arg[6]
        x = np.array([x_0, x_1, x_2])
        norm = np.linalg.norm(x)
        x /= max(norm,1e-10)
        #v = omega* x - H @ x + 1.j* (beta * np.conj(x) @ x) * x
        v = omega* x - H @ x + arg[1]*1.j* (beta * np.abs(x)**2) * x        
        loss = np.conj(v).T @ v
        return loss.real
        
    npr_eig = np.zeros((npr_kappa.shape[0], n), dtype=float)
    npr_eig_org = np.zeros((npr_kappa.shape[0], n), dtype=float)
    npr_fun = np.zeros((npr_kappa.shape[0], n), dtype=float)
    npr_amp = np.zeros((npr_kappa.shape[0], n), dtype=float)

    for i, kappa in enumerate(npr_kappa):  
        cls_rr = rrarray.RRarray(n, delta, npr_Delta, npr_eta, kappa, theta, kappa2, theta2, savefig=True, boundary="open")
        def loss_opt(arg):
            return loss(cls_rr.H, arg)

        eigval, eigvec = cls_rr.solve()

        for j, eig in enumerate(eigval):
            if eig.imag < 0:
                npr_eig[i,j] = np.nan#eig.real
                npr_eig_org[i,j] = eig.real
                npr_fun[i,j] = np.inf
                continue

            amp_theo = np.sqrt(np.abs(eig.imag+gain) / gs)
            vec = eigvec[:,j]
            vec *= np.exp(-1.j*np.angle(vec[0]))
            arg = np.array([vec[0].real, vec[1].real, vec[1].imag, vec[2].real, vec[2].imag])
            x0 = np.array([eig.real, amp_theo**2]+ arg.tolist()).flatten()
            #x0 = np.array([eig] + [amp_theo, amp_theo, 0, amp_theo,0])

            bounds = [(0, None)] + [(None, None)] + [(None, None)]*5
            res = minimize(loss_opt, x0, tol=1e-20, options={"maxiter": 1000}, bounds=bounds)

            npr_fun[i,j] = res.fun
            if res.fun < 1e-10:
                npr_eig[i,j] = res.x[0]
                npr_amp[i,j] = res.x[1]
            else:
                npr_eig[i,j] = np.nan
                npr_amp[i,j] = np.nan

            npr_eig_org[i,j] = eig.real

    print(npr_fun)

    # plot
    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111)
    for j in range(n):
        ax.scatter(npr_kappa, npr_eig[:,j], label=f"$\omega_{j}$", color="blue")
        ax.scatter(npr_kappa, npr_eig_org[:,j], label=f"$\omega_{j}$", color="gray",marker="x")
    ax.set_xlabel("$\kappa$")
    ax.set_ylabel("$\omega$")
    plt.tight_layout()
    plt.show()

# %%

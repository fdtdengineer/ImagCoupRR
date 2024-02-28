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
    gain = 1
    gs = 1e-3

    delta = 0.0
    npr_Delta = np.array([1,2,4])#+10
    
    n = npr_Delta.shape[0]
    npr_eta = np.ones(n)*gain
    kappa = 0.5 #
    theta = np.pi #0.2*np.pi #
    kappa2 = 0 #
    theta2 = 0 #

    npr_kappa = np.linspace(0, 2, 201)    
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
    cme = time_evolution.CoupledModeEquation(cls_rr2.H, dt=0.01, tmax=500)
    cme.set_initial_state(a0)
    cme.solve_stuartlandau(beta=gs)
    
    cme.plot("psiReal")
    cme.plot("psiAbs")
    cme.plot("psiAbsRel")

    ave_r = cme.get_average(key="psiReal", num_data=1)
    ave_rAbs = cme.get_average(key="psiAbs", num_data=1)
    ave_rRel = cme.get_average(key="psiAbsRel", num_data=1)
    ave_phi = cme.get_average(key="psiPhaseRel", num_data=1)
    
    dict_fft = cme.get_fft(num_data=25000)
    df_fft = dict_fft["df"]
    peak = dict_fft["peak"]
    decay = dict_fft["decay"]

    plt.plot(df_fft)
    plt.xlim(0, 5)

    print(peak)
    #print("end")

    # Time-simulated solution
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
    print("\n")

    # Analytical solution (linear)
    eigval, eigvec = np.linalg.eig(cls_rr2.H)
    idx_minloss = eigval.imag.argmax()
    #print("eigval, eigvec")
    eval = eigval[idx_minloss]
    evec = eigvec[:, idx_minloss]
    print("Analytical solution (linear)")
    print(eval * evec)
    print(cls_rr2.H @ evec)
    print("\n")
    print(eval * phi0)
    amp_theo = np.sqrt(np.abs(eval.imag) / gs)
    #print(eigval[idx_minloss])
    #print(eigvec[:, idx_minloss])

    #%%

    #########################################
    ###### Calibration ######################
    #########################################
    from scipy.optimize import minimize

    def loss(H, arg, beta=1e-3):
        omega = arg[0] #+ 1.j * arg[1]
        x_0 = arg[1] #+ 1.j * arg[2]
        x_1 = arg[2] + 1.j * arg[3]
        x_2 = arg[4] + 1.j * arg[5]
        x = np.array([x_0, x_1, x_2])
        #x /= np.linalg.norm(x)

        v = omega* x - H @ x + 1.j* beta * np.conj(x) @ x * x
        loss = np.conj(v).T @ v
        #loss += np.linalg.norm(np.abs(x) - amp_theo)**2
        return loss.real
    
    def loss_opt(arg):
        return loss(cls_rr2.H, arg)

    x0 = np.array([3] + [amp_theo, amp_theo, 0, amp_theo,0])
    #x0 = np.array([
    #        peak[idx_sol], phi[0].real, phi[1].real, phi[1].imag, phi[2].real, phi[2].imag
    #    ])
    bounds = [(0, None)] + [(None, None)]*5
    res = minimize(loss_opt, x0, tol=1e-20, options={"maxiter": 1000}, bounds=bounds)
    print(res)
    
    """
    def gauss_newton(f, x0, eps=1e-4, tol=1e-20, maxiter=1000):
        x = x0
        nx = x.shape[0]
        nf = f(x).shape[0]
        for i in range(maxiter):
            J = np.zeros((nf, nx))
            for j in range(n):
                xh = x.copy()
                xl = x.copy()
                xh[j] += eps
                xl[j] -= eps
                J[:, j] = (f(xh) - f(xl)) / (2 * eps)
            dx = np.linalg.pinv(J) @ f(x)
            x -= dx
            if np.linalg.norm(dx) < tol:
                break
        return x

    x = gauss_newton(loss_opt, x0)
    print(x)
    # check
    print(loss_opt(x))
    """
    

# %%

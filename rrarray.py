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
    from mpl_toolkits.mplot3d import Axes3D
    #import ipywidgets

    filepath_output = "output\\"


class RRarray:
    def __init__(self, n, omega, npr_Delta, npr_eta, kappa, theta, kappa2, theta2, savefig=False, boundary="open", flg_flux=False):
        """
        n: number of Ring Resonators
        omega: detuning of the ring resonators 
        npr_delta (vector): refractive index change of the ring
        npr_eta (vector): extinction coefficient change of the ring
        kappa: coupling coefficient
        theta: phase shift between the two ring resonators
        kappa2: coupling coefficient (next nearest neighbor)
        theta2: phase shift between the two ring resonators (next nearest neighbor)
        """
        self.dict_vals = {
            "n":n, 
            "omega":omega, 
            "npr_delta":npr_Delta, 
            "npr_eta":npr_eta, 
            "kappa":kappa, 
            "theta":theta,
            "kappa2":kappa2, 
            "theta2":theta2,
            "boundary":boundary,
            }
        self.savefig = savefig
        #self.n = n
        self.filepath_output = "fig\\"
        self.flg_flux = flg_flux
        if n<=2:
            Exception("The number of ring resonators must be greater than 2.")
        elif len(npr_Delta) != n:
            Exception("The length of npr_delta must be equal to n.")
        elif len(npr_eta) != n:
            Exception("The length of npr_eta must be equal to n.")
        else:
            pass

        self._build_hamiltonian()

    def _build_hamiltonian(self, dict_keys={}):
        for key, val in dict_keys.items():
            self.dict_vals[key] = val
            #self.dict_vals["kappa2"] = self.dict_vals["kappa"]
        
        self.n = self.dict_vals["n"]
        self.omega = self.dict_vals["omega"]
        self.npr_delta = self.dict_vals["npr_delta"]
        self.npr_eta = self.dict_vals["npr_eta"]
        self.kappa = self.dict_vals["kappa"]
        self.theta = self.dict_vals["theta"]
        self.kappa2 = self.dict_vals["kappa2"]
        self.theta2 = self.dict_vals["theta2"]
        self.boundary = self.dict_vals["boundary"]

        self.H = np.zeros((self.n, self.n))
        self.H = self.H.astype(complex)

        arg = np.exp(1j * self.theta)

        # diagonal elements
        if self.boundary=="open":
            self.H[0, 0] = -self.omega + self.npr_delta[0] + 1j * self.npr_eta[0] - 2j * self.kappa
            self.H[self.n-1, self.n-1] = -self.omega + self.npr_delta[self.n-1] + 1j * self.npr_eta[self.n-1] - 2j * self.kappa
        elif self.boundary=="periodic":
            self.H[0, 0] = -self.omega + self.npr_delta[0] + 1j * self.npr_eta[0] - 4j * self.kappa
            self.H[self.n-1, self.n-1] = -self.omega + self.npr_delta[self.n-1] + 1j * self.npr_eta[self.n-1] - 4j * self.kappa
            self.H[0, self.n-1] = (-2j) * self.kappa * arg
            self.H[self.n-1, 0] = (-2j) * self.kappa * arg
        for i in range(1, self.n-1):
            self.H[i, i] = -self.omega + self.npr_delta[i] + 1j * self.npr_eta[i] - 4j * self.kappa 

        # off-diagonal elements
        H_od1 = np.diag(np.ones(self.n-1),1)
        H_od2 = np.diag(np.ones(self.n-1),1).T

        H_od1 = H_od1 * (-2j) * self.kappa * arg
        H_od2 = H_od2 * (-2j) * self.kappa * arg

        # next nearest neighbor
        Hnnn_od1 = np.diag(np.ones(self.n-2),2)
        Hnnn_od2 = np.diag(np.ones(self.n-2),2).T

        arg_nnn = np.exp(1j * self.theta2)
        Hnnn_od1 = Hnnn_od1 * (-2j) * self.kappa2 * arg_nnn
        Hnnn_od2 = Hnnn_od2 * (-2j) * self.kappa2 * arg_nnn

        self.H += H_od1 + H_od2 + Hnnn_od1 + Hnnn_od2


        if self.flg_flux:
            self.H_inv = np.linalg.pinv(self.H)
        else:
            # Dummy identity matrix
            self.H_inv = np.eye(self.n)
        return self.H

    def get_Hamiltonian(self):
        return self.H

    def solve(self):
        eigdat = np.linalg.eig(self.H)
        self.eigval = eigdat[0]
        self.eigvec = eigdat[1]
        return self.eigval, self.eigvec
    
    def sweep(self, npr_list, list_keys=["kappa"], str_cmap="viridis"):
        self.npr_list = npr_list
        n_point = npr_list.shape[1]
        self.eigval_list = np.zeros((n_point, self.n)).astype(complex)
        self.eigvec_list = np.zeros((n_point, self.n, self.n)).astype(complex)

        for i in range(n_point):
            dict_keys = {}
            for j, key in enumerate(list_keys):
                dict_keys[key] = npr_list[j][i]

            self._build_hamiltonian(dict_keys=dict_keys)
            eigval, eigvec = self.solve()
            
            # sort by imaginary part
            # idx_sort = eigval.real.argsort()
            #eigval = eigval[idx_sort]
            #eigvec = eigvec[:, idx_sort]            
            
            self.eigval_list[i] = eigval
            self.eigvec_list[i] = eigvec
                     
        n_state = self.eigval_list.shape[1]
        cmap = cm.get_cmap(str_cmap)
        self.list_color = cmap(np.linspace(0, 1, n_state))

        if key=="theta":
            self.npr_list /= np.pi

        eigval_list_real = self.eigval_list.real.astype(float)
        df_eigval_r = pd.DataFrame(eigval_list_real)
        if np.ndim(npr_list) < 2:
            df_eigval_r.index = self.npr_list[0]
        df_eigval_r.index.name = list_keys[0]
        df_eigval_r.columns = ["$\omega_{%d}$"%i for i in range(n_state)]
        self.df_eigval_r = df_eigval_r
        self.df_eigval_r.to_csv(self.filepath_output + "eigval_real.csv")

    def plot_eigval_sweep(
            self, 
            xlabel="$\kappa'$", 
            show_label=True, 
            grid=False, 
            list_ylim=[-6, 1],
            list_figsize=[6,3],
            idx_key=0, 
            npr_x=np.array([])
            ):
        ### plot Re and Im in two figures
        fig = plt.figure(figsize=(list_figsize))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        if len(npr_x)==0:
            npr_x = self.npr_list[idx_key]
        for i in range(self.n):
            #ax1.plot(self.npr_list, self.eigval_list[:,i].real, "o", label="Re$(\omega_{%d})$"%i, color=self.list_color[i])
            #ax2.plot(self.npr_list, self.eigval_list[:,i].imag, "o", label="Im$(\omega_{%d})$"%i, color=self.list_color[i])
            ax1.plot(npr_x, self.eigval_list[:,i].real, "o", color=self.list_color[i])
            ax2.plot(npr_x, self.eigval_list[:,i].imag, "o", label="$\omega_{%d}$"%i, color=self.list_color[i])

        ax1.set_xlabel(xlabel)
        ax1.set_ylabel("Re$(\omega)$")
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel("Im$(\omega)$")
        if len(list_ylim)==2:
            ax2.set_ylim(list_ylim)
        #ax1.legend(fontsize=fs*0.5)
        if show_label:
            # ax2.legend を ax2 の外に出す　右側に配置
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=fs*0.6)
            #ax2.legend(fontsize=fs*0.5)
        if grid:
            ax1.grid()
            ax2.grid()
        plt.tight_layout()
        if self.savefig:
            plt.savefig(self.filepath_output + "omega.svg", transparent=True)
        plt.show()

    def show_eigvec(self, idx):
        norm = np.conj(self.eigvec_list[idx]) * self.eigvec_list[idx]
        
        n_site = norm.shape[0]
        n_state = norm.shape[1]

        # cmap をインポート
        # グラフを5つ並べる
        fig, axes = plt.subplots(nrows=n_state, ncols=1, figsize=(4,6))
        for i in range(n_state):
            axes[i].bar(np.arange(n_site), norm[:,i], label="state %d"%i, color=self.list_color[i])
            axes[i].set_ylim(0, 1)
            #axes[i].legend(fontsize=fs*0.5)
            # x軸の目盛りを非表示
            if i!=n_state-1:
                axes[i].tick_params(labelbottom=False)
            axes[i].set_ylabel("${|u_"+str(i)+"|}^2$")

        plt.tight_layout()
        if self.savefig:
            plt.savefig(self.filepath_output + "density.svg", transparent=True)
        plt.show()

    # flux
    def solve_flux(self, npr_freq, s1=1, idx=-1):
        self.npr_freq = npr_freq

        if idx==-1:
            idx = self.n - 1
        npr_input = np.zeros(self.n, dtype=complex)
        npr_input[0] = 1.j*np.sqrt(2*self.npr_eta[0])*s1
 
        self.npr_R = np.zeros(self.npr_freq.shape[0], dtype=complex)
        self.npr_T = np.zeros(self.npr_freq.shape[0], dtype=complex)
 
        for i, freq in enumerate(npr_freq):
            self._build_hamiltonian(dict_keys={"omega":freq})
            npr_amp = np.dot(self.H_inv, npr_input)
            a1 = npr_amp[0]
            an = npr_amp[idx]
            self.npr_R[i] = np.abs(-1 + np.sqrt(2*self.npr_eta[0]) * a1 / s1)**2
            self.npr_T[i] = 2*self.npr_eta[-1] * np.abs(an / s1)**2

        # real float
        self.npr_R = self.npr_R.real.astype(float)
        self.npr_T = self.npr_T.real.astype(float)
        
        self.df_flux = pd.DataFrame({"freq":self.npr_freq.real, "R":self.npr_R, "T":self.npr_T})
        self.df_flux.to_csv(self.filepath_output + "flux.csv", index=False)

    def plot_flux(self, plottype="T"):
        """
        plot the amplitude of the output port for each input frequency
        """
        fig = plt.figure(figsize=(3,3))
        ax = fig.add_subplot(111)
        if plottype=="T":
            ax.plot(self.npr_freq, self.npr_T, label="T", color="teal")
        elif plottype=="R":
            ax.plot(self.npr_freq, self.npr_R, label="R", color="gray")


        ax.set_xlabel(r"$\omega$")
        ax.set_ylabel(plottype)
        ax.legend(fontsize=fs*0.6)
        #ax.set_ylim(0, 1)
        plt.tight_layout()
        if self.savefig:
            plt.savefig(self.filepath_output + "flux.svg", transparent=True)
        plt.show()

    def fit_by_lorentzian(self, xmin=-0.5,xmax=0.5, plottype="T",p0=[1, 0, 1, 2]):
        x = self.npr_freq
        if plottype=="T":
            y = self.npr_T
        elif plottype=="R":
            y = self.npr_R
        else:
            Exception("plottype must be 'T' or 'R'.")

        mask = (x>xmin)*(x<xmax)
        x = x[mask]
        y = y[mask]
        
        from scipy.optimize import curve_fit
        def func(x, a, b, c, d):
            return a / ((x-b)**2 + c)**d
        
        popt, pcov = curve_fit(func, x, y, p0=p0)
        print("a: ", popt[0])
        print("b: ", popt[1])
        print("c: ", popt[2])
        print("d: ", popt[-1])
        d_lorentz = popt[-1]

        y_fit = func(x, *popt)
        fig = plt.figure(figsize=(3,3))
        ax = fig.add_subplot(111)
        ax.plot(x, y, label=plottype, color="gray")
        ax.plot(x, y_fit, label="fit", color="teal", linestyle="dashed")
        ax.set_xlabel(r"$\omega$")
        ax.set_ylabel(plottype)
        ax.text(0.5, 0.5, "$d$ = %.2f"%d_lorentz, transform=ax.transAxes, fontsize=fs*0.8)
        ax.legend(fontsize=fs*0.6)
        #ax.set_ylim(0, 1)
        plt.tight_layout()
        plt.show()



if __name__ == "__main__":
    delta=0
    #npr_Delta = np.array([-2,2,-1,0,1])
    npr_Delta = np.array([-2,-1,0,1,2])
    #npr_Delta = np.array([-5,-1,0,1,5])
    #npr_Delta = -np.array([-6,0,3])
    #npr_Delta = np.zeros(5)
    n = npr_Delta.shape[0]

    #npr_Delta = np.zeros(5)
    #n = npr_Delta.shape[0]
    #npr_eta = np.zeros(n) #-0#.1
    npr_eta = np.array([1, 0., 0., 0., 1])*0.5

    kappa = 1#*(-1j) # dummy because of the sweep
    theta = np.pi
    kappa2 = 0#0.5*(-1j) #無効
    theta2 = 0 # 0.1*np.pi
    boundary = "open" #"periodic"

    #npr_kappa = np.linspace(0, 10, 200)
    npr_kappa = np.linspace(0, 2, 200)
    
    npr_kappa2 = npr_kappa#*0 + np.zeros(npr_kappa.shape[0])
    npr_sweep = np.array([npr_kappa,npr_kappa2]) # kappa と kappa2 を同時に変化させる

    cls_rr = RRarray(n, delta, npr_Delta, npr_eta, kappa, theta, kappa2, theta2, savefig=True, boundary=boundary)
    cls_rr.sweep(npr_sweep, list_keys=["kappa","kappa2"])
    cls_rr.plot_eigval_sweep()
    cls_rr.show_eigvec(npr_kappa.shape[0] - 1)

    print("Done.")






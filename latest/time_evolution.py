

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

    filepath_output = "fig\\"


class CoupledModeEquation:
    # class for the coupled mode equation by Runge-Kutta method
    def __init__(self, H, dt=0.1, tmax=100):
        # H: Hamiltonian as np.array
        self.H = H
        self.N = H.shape[0]
        self.dt = dt
        self.t = 0
        self.tmax = tmax
        self.Nt = int(self.tmax/self.dt)
        self.tlist = np.linspace(0, self.tmax, self.Nt)
        self.psi = np.zeros((self.N, self.Nt), dtype=np.complex)

        self.dict_results = {}
        self.dict_labels = {"psiReal": "$\psi_r$", "psiImag": "$\psi_i$", "psiAbs": "$\|\psi\|$", "psiAbsRel": "$\|\psi\| / \|\psi_0\|$", "psiPhase": "$\phi$", "psiPhaseRel": "$\phi - \phi_0$"}

    def set_initial_state(self, psi0):
        self.psi[:, 0] = psi0
    
    def solve(self):
        iH = 1j*self.H
        for i in range(self.Nt-1):
            k1 = -iH @ self.psi[:, i]
            k2 = -iH @ (self.psi[:, i] + self.dt/2*k1)
            k3 = -iH @ (self.psi[:, i] + self.dt/2*k2)
            k4 = -iH @ (self.psi[:, i] + self.dt*k3)
            self.psi[:, i+1] = self.psi[:, i] + self.dt/6*(k1 + 2*k2 + 2*k3 + k4)
        self.t += self.tmax

        self.dict_results["psiReal"] = self.psi.real
        self.dict_results["psiImag"] = self.psi.imag
        self.dict_results["psiAbs"] = np.abs(self.psi)
        self.dict_results["psiAbsRel"] = np.abs(self.psi) / np.abs(self.psi[0])

        # 0 to 2
        self.dict_results["psiPhase"] = np.angle(self.psi) / np.pi - 1
        self.dict_results["psiPhaseRel"] = ((np.angle(self.psi) - np.angle(self.psi[0])) / np.pi - 1) % 2 - 1


    def plot(self, key="psiReal", color="gray"):
        fig = plt.figure(figsize=(4, 3))
        ax = fig.add_subplot(111)
        
        npr_y = self.dict_results[key]

        for i in range(self.N):
            if i == 0:
                color_i = "black"
            else:
                color_i = color
            ax.plot(self.tlist, npr_y[i, :], label=self.dict_labels[key] + f"$_{i+1}$", color=color_i)

        ax.set_xlabel("$t$")
        ax.set_ylabel(self.dict_labels[key])
        
        ax.set_xlim(0, self.tmax)
        if key == "psiPhaseRel":
            ax.set_ylim(-1, 1)

        #ax.legend()
        plt.tight_layout()
        plt.savefig(filepath_output + "cme_" + key + ".png", dpi=300)
        plt.show()
        plt.close()



if __name__ == "__main__":
    fc = 1 - 0.5j
    f1 = 1
    f2 = 1.01
    g = 0.1

    H = np.array(
        [[f1,  g, 0],
         [ g, fc, g],
         [ 0,  g, f2]]
         )
    
    #psi0 = np.array([1, 1, 1], dtype=np.complex)
    psi0 = np.array([1, 0, 0], dtype=np.complex)
    #psi0*=30
    cme = CoupledModeEquation(H)
    cme.set_initial_state(psi0)
    cme.solve()
    cme.plot("psiAbs")

    print("finished")

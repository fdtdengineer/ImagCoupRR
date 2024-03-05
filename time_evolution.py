#%%
if True:
    import numpy as np
    import pandas as pd
    from scipy.fftpack import fft
    from scipy.signal import find_peaks
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
    # linear fit
    def linfit(x, y):
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        return m, c

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
        self.psi = np.zeros((self.N, self.Nt), dtype=complex)

        self.dict_results = {}
        self.dict_labels = {"psiReal": "$Re(a_i)$", "psiRealRel": "$Re(a_i)/r_i$", "psiImag": "$Im(a_i)$", "psiAbs": "$r_i$", "psiAbsRel": "$r_i / r_0$", "psiPhase": "$\phi$", "psiPhaseRel": "$\phi_i - \phi_0$"}

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

    def solve_stuartlandau(self, beta=1e-3):
        iH = 1j*self.H
        for i in range(self.Nt-1):
            k1 = -iH @ self.psi[:, i] - beta * np.abs(self.psi[:, i])**2 * self.psi[:, i]
            psi_k1 = self.psi[:, i] + self.dt/2*k1
            k2 = -iH @ psi_k1 - beta * np.abs(psi_k1)**2 * psi_k1
            psi_k2 = self.psi[:, i] + self.dt/2*k2
            k3 = -iH @ psi_k2 - beta * np.abs(psi_k2)**2 * psi_k2
            psi_k3 = self.psi[:, i] + self.dt*k3
            k4 = -iH @ psi_k3 - beta * np.abs(psi_k3)**2 * psi_k3

            self.psi[:, i+1] = self.psi[:, i] + self.dt/6*(k1 + 2*k2 + 2*k3 + k4)
        self.t += self.tmax

        self.dict_results["psiReal"] = self.psi.real
        self.dict_results["psiImag"] = self.psi.imag
        self.dict_results["psiRealRel"] = self.psi.real / np.abs(self.psi[0])

        self.dict_results["psiAbs"] = np.abs(self.psi)
        self.dict_results["psiAbsRel"] = np.abs(self.psi) / np.abs(self.psi[0])

        # 0 to 2
        self.dict_results["psiPhase"] = np.angle(self.psi) / np.pi - 1
        self.dict_results["psiPhaseRel"] = ((np.angle(self.psi) - np.angle(self.psi[0])) / np.pi - 1) % 2 - 1

    def solve_3rdwnorm(self, beta=1e-3):
        iH = 1j*self.H
        for i in range(self.Nt-1):
            k1 = -iH @ self.psi[:, i] - beta * (np.conj(self.psi[:, i]) @ self.psi[:, i]) * self.psi[:, i]
            psi_k1 = self.psi[:, i] + self.dt/2*k1
            k2 = -iH @ psi_k1 - beta * (np.conj(psi_k1) @ psi_k1) * psi_k1
            psi_k2 = self.psi[:, i] + self.dt/2*k2
            k3 = -iH @ psi_k2 - beta * (np.conj(psi_k2) @ psi_k2) * psi_k2
            psi_k3 = self.psi[:, i] + self.dt*k3
            k4 = -iH @ psi_k3 - beta * (np.conj(psi_k3) @ psi_k3) * psi_k3

            self.psi[:, i+1] = self.psi[:, i] + self.dt/6*(k1 + 2*k2 + 2*k3 + k4)
        self.t += self.tmax

        self.dict_results["psiReal"] = self.psi.real
        self.dict_results["psiImag"] = self.psi.imag
        self.dict_results["psiRealRel"] = self.psi.real / np.abs(self.psi[0])

        self.dict_results["psiAbs"] = np.abs(self.psi)
        self.dict_results["psiAbsRel"] = np.abs(self.psi) / np.abs(self.psi[0])

        # 0 to 2
        self.dict_results["psiPhase"] = np.angle(self.psi) / np.pi - 1
        self.dict_results["psiPhaseRel"] = ((np.angle(self.psi) - np.angle(self.psi[0])) / np.pi - 1) % 2 - 1

    def get_average(self, key="psiReal", num_data=30):
        # get average of the last num_data
        return np.mean(self.dict_results[key][:, -num_data:], axis=1)

    def plot(self, key="psiReal", color="gray", list_ylim=[], yscale="linear"):
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
        if len(list_ylim) == 2:
            ax.set_ylim(list_ylim)

        #ax.legend()
        plt.tight_layout()
        if yscale == "log":
            ax.set_yscale("log")
        plt.savefig(filepath_output + "cme_" + key + ".svg", transparent=True)
        plt.show()
        plt.close()

    def save_csv(self, key="psiReal", filename="cme.csv"):
        df = pd.DataFrame(
            self.dict_results[key].T, 
            columns=[key+f"{i+1}" for i in range(self.N)],
            index=self.tlist
            )
        df.index.name = "t"
        df.to_csv(filepath_output + filename)
    

    def get_fft(self, num_data=1000, type_peak="max"):
        key_re="psiRealRel"
        key_im="psiAbs"

        dict_fft = {}
        freq = 2*np.pi*np.fft.fftfreq(num_data, self.dt)
        freq = freq[:int(num_data/2)]

        def get_peak(npr_abs_y_fft, freq, height=0.03):
            if type_peak == "max":
                idx_peak = np.argmax(npr_abs_y_fft)
                peak = freq[idx_peak]
                return peak
            elif type_peak == "all":
                peaks, _ = find_peaks(npr_abs_y_fft, height=height)
                return freq[peaks]
            else:
                Exception("type_peak should be max or all")

        npr_peak = np.zeros(self.N)
        list_peak = []
        npr_decay = np.zeros(self.N)
        
        for idx in range(self.N):
            npr_y = self.dict_results[key_re]
            npr_y = npr_y[idx, -num_data:]
            npr_y_fft = fft(npr_y)/(num_data/2)
            npr_y_fft_positive = np.abs(npr_y_fft)[:int(num_data/2)]
            dict_fft[f"{key_re}_{idx}"] = npr_y_fft_positive

            # peak frequency
            if type_peak == "max":
                npr_peak[idx] = get_peak(npr_y_fft_positive, freq)
            elif type_peak == "all":
                list_peak.append(get_peak(npr_y_fft_positive, freq).tolist())
            else:
                Exception("type_peak should be max or all")

            # exp decay rate 
            npr_r = self.dict_results[key_im]
            npr_r = npr_r[idx, -num_data:]
            npr_t = self.tlist[-num_data:]
            m, c = linfit(npr_t, np.log(npr_r) )
            npr_decay[idx] = m
             
        if type_peak == "all":
            list_peak = [item for sublist in list_peak for item in sublist]
            list_peak = list(set(list_peak))
            npr_peak = np.array(list_peak)

            #peak = peak[peak <= 4.01]
            #peak = peak[peak >= 0.99]
            # fill np.nan to be the lenght =3 if len(npr_peak)<3
            if len(npr_peak) < 3:
                npr_peak = np.concatenate([npr_peak, np.nan*np.zeros(3-len(npr_peak))])
            
        df_fft = pd.DataFrame(dict_fft, index=freq)
        return {"df": df_fft, "peak": npr_peak, "decay": npr_decay}




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
    psi0 = np.array([1, 0, 0], dtype=np.complex128)
    #psi0*=30
    cme = CoupledModeEquation(H, dt=0.01, tmax=10)
    cme.set_initial_state(psi0)
    cme.solve()
    #cme.plot("psiAbs")
    cme.plot("psiAbsRel")
    cme.plot("psiPhaseRel")
    cme.plot("psiPhase")    
    average = cme.get_average(key="psiPhaseRel", num_data=30)

    print(average)
    print("finished")

# %%

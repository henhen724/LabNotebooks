import matplotlib.pyplot as plt

GPUAcc = False
try:
    import cupy as xp
    GPUAcc = True
    print("Using GPU Acceleration")
    import numpy as np
except:
    import numpy as xp
    import numpy as np
import time

def check_CuPy(arr):
    if GPUAcc and isinstance(arr, np.ndarray):
        return xp.array(arr)
    return arr


class ParrallelTempering:
    def __init__(self, Jnonlocal, Jlocal, K, Zenergy, temps, Nrepl=500, Nspins=20, steps=10000, sigma=xp.pi/10, sigmaR=0.01, confine_energy_mag=1e12, Equilibration=100, Nbins=79):
        self.Jnonlocal = check_CuPy(Jnonlocal)
        self.Jlocal = check_CuPy(Jlocal)
        self.K = check_CuPy(K)
        assert(self.Jnonlocal.shape[0] == self.Jlocal.shape[0] and self.Jnonlocal.shape == self.K.shape)
        self.Zenergy = Zenergy
        self.temps = temps
        self.Equilibration=Equilibration
        self.Nrepl = Nrepl
        self.Ntemps = temps.shape[0]
        self.Nspins = Nspins
        self.steps = steps
        self.sigma = sigma
        self.sigmaR = sigmaR
        self.confine_energy_mag = confine_energy_mag
        self.energy_record = xp.zeros((self.Ntemps, self.Nrepl, self.steps))
        self.flip_acc_rec = xp.zeros(self.steps//self.Equilibration)
        self.angle_overlap_hist = xp.zeros((self.Ntemps, Nbins, Nbins)) # axis one is Q = qxx + qyy and axis two is R = qxx - qyy
        self.radial_hist = xp.zeros((self.Ntemps, 50))
        self.rho_hist = xp.zeros((self.Ntemps, 50))
        self.curr_state = xp.zeros((self.Ntemps, self.Nrepl, self.Nspins, 3))

    def run(self):
        # indexing for the states
        self.curr_state[:, :, :, 0] = xp.ones((self.Nrepl, self.Nspins))
        self.curr_state[:, :, :, 1] = xp.pi * \
            xp.random.rand(self.Nrepl, self.Nspins)
        self.curr_state[:, :, :, 2] = 2*xp.pi * \
            xp.random.rand(self.Nrepl, self.Nspins)

        displacement = xp.zeros((self.Ntemps, self.Nrepl, self.Nspins, 3))
        for eqlNum in range(self.steps//self.Equilibration):
            # run metropolis for 1 equilibration time
            for i in range(self.Equilibration):
                # print("Step: ", i)
                self.energy_record[:, :, i] = self.motional_model(self.curr_state)
                # angular displacements
                if xp.random.rand() < 0.5:
                    displacement[:, :, :, 0] = xp.zeros((self.Ntemps, self.Nrepl, self.Nspins))
                    displacement[:, :, :, 1:] = xp.random.normal(
                        loc=0.0, scale=self.sigma, size=(self.Ntemps, self.Nrepl, self.Nspins, 2))
                else:
                    displacement[:, :, :, 0] = xp.random.normal(
                        loc=0.0, scale=self.sigmaR, size=(self.Ntemps, self.Nrepl, self.Nspins))
                    displacement[:, :, :, 1:] = xp.zeros((self.Ntemps, self.Nrepl, self.Nspins, 2))

                tentative_state = displacement + self.curr_state

                # print("Enegies NAN: ")
                # print(xp.isnan(confine_energy(tentative_state)))
                # print(xp.isnan(confine_energy(self.curr_state[:,i])))
                # print(xp.isnan(motional_model(tentative_state, Zenergy, Jlocal, self.Jnonlocal)))
                # print(xp.isnan(motional_model(self.curr_state[:,i], Zenergy, Jlocal, self.Jnonlocal)))
                # print("\n")

                DeltaE = self.confine_energy(tentative_state) - self.confine_energy(self.curr_state) + self.motional_model(
                    tentative_state) - self.motional_model(self.curr_state)

                # print("DE: ", DeltaE[0], "R before", self.curr_state[0,:,0], "R after", tentative_state[0,:,0])

                state_decissions = xp.logical_or(
                    (DeltaE <= 0), (xp.exp(-xp.einsum("ab,a->ab", xp.abs(DeltaE), 1/self.temps)) > xp.random.rand(1)))
                # print("Decisions: ", state_decissions[0])
                # print(tentative_state.shape, self.curr_state.shape)
                self.curr_state = xp.einsum("fi,fijk->fijk", state_decissions, tentative_state) + xp.einsum(
                    "fi,fijk->fijk", xp.logical_not(state_decissions), self.curr_state)
                
            self.record_angle_overlap()
            self.record_radial_hist()
            self.record_rho_hist()

            # Now swap neighbouring tempuratures based on total energy
            curr_energies = self.motional_model(self.curr_state)
            mod2Compare = eqlNum % 2 # switch between comparing 0 mod 4 to 1 mod 4 and comparing 1 mod 4 to 2 mod 4
            numOfFlips = 0
            for comparisonNum in range(self.Ntemps//2 - 1):
                lowerTemp = 2*comparisonNum+mod2Compare
                higherTemp = 2*comparisonNum+mod2Compare+1
                state_decissions = curr_energies[higherTemp] < curr_energies[lowerTemp]
                temporarylowerTempStates = xp.einsum("i,ijk->ijk", state_decissions, self.curr_state[higherTemp]) + xp.einsum(
                    "i,ijk->ijk", xp.logical_not(state_decissions), self.curr_state[lowerTemp])
                self.curr_state[higherTemp] = xp.einsum("i,ijk->ijk", state_decissions, self.curr_state[lowerTemp]) + xp.einsum(
                    "i,ijk->ijk", xp.logical_not(state_decissions), self.curr_state[higherTemp])
                self.curr_state[lowerTemp] = temporarylowerTempStates
                numOfFlips += xp.sum(state_decissions)
            print("Fraction of Flips Accepted: ", numOfFlips/(self.Nrepl*(self.Ntemps//2)))
            self.flip_acc_rec[eqlNum] = numOfFlips/(self.Nrepl*(self.Ntemps//2))

        self.energy_record[:, :, self.steps -
                           1] = self.motional_model(self.curr_state)

        return self.curr_state

    def motional_model(self, this_state):
        energy = xp.zeros((self.Ntemps, self.Nrepl))
        energy += -self.Zenergy * \
            xp.sum(this_state[:, :, :, 0]*xp.cos(this_state[:, :, :, 1]), axis=2)
        if xp.isnan(energy).any():
            print("Zenergy threw the nan")
        x_vec = this_state[:, :, :, 0] * \
            xp.sin(this_state[:, :, :, 1])*xp.cos(this_state[:, :, :, 2])
        y_vec = this_state[:, :, :, 0] * \
            xp.sin(this_state[:, :, :, 1])*xp.sin(this_state[:, :, :, 2])

        rho = xp.square(x_vec) + xp.square(y_vec)

        energy += -xp.einsum("i,kji->kj", self.Jlocal, rho)
        if xp.isnan(energy).any():
            print("Jlocal threw the nan")
        energy += -xp.einsum("afj,jk,afk->af", x_vec, self.Jnonlocal, x_vec) + \
            xp.einsum("afj,jk,afk->af", y_vec, self.Jnonlocal, y_vec)
        if xp.isnan(energy).any():
            print("self.Jnonlocal threw the nan")
        energy += -xp.einsum("afj, jk, afk->af", x_vec, self.K, y_vec) - \
            xp.einsum("afj,jk,afk->af", y_vec, self.Jnonlocal, x_vec)
        if xp.isnan(energy).any():
            print("self.K threw the nan")
        if xp.isnan(xp.einsum("afj,jk,afk->f", x_vec, self.Jnonlocal, x_vec)).any():
            print("X is: ", x_vec, "\n\n R is:",
                  this_state[:, :, :, 0], "\n\n theta is:", this_state[:, :, :, 1], "\n\n phi is:", this_state[:, :, :, 2])
        return energy

    def confine_energy(self, this_state):
        energy = xp.zeros((self.Ntemps, self.Nrepl))
        # print("State: ", state)
        for i in range(self.Nspins):
            spin = this_state[:, :, i, :]
            energy += self.confine_energy_mag * \
                xp.abs(xp.logical_or(spin[:, :, 0] >= 1,
                       spin[:, :, 0] < 0))*xp.abs(spin[:, :, 0])
        return energy

    def plot_energy_record(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(xp.arange(self.steps), self.energy_record.get().T)

    def plot_single_spin_hist(self, spin, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        x_vec = self.curr_state[:, :, :, 0] * \
            xp.sin(self.curr_state[:, :, :, 1]) * \
            xp.cos(self.curr_state[:, :, :, 2])
        y_vec = self.curr_state[:, :, :, 0] * \
            xp.sin(self.curr_state[:, :, :, 1]) * \
            xp.sin(self.curr_state[:, :, :, 2])
        if GPUAcc:
            ax.hist2d(x_vec[:, :, spin].get(),
                      y_vec[:, :, spin].get(), bins=100)
        else:
            ax.hist2d(x_vec[:, :, spin], y_vec[:, spin], bins=100)

    def f_rd(self, M):
        return xp.ndarray.flatten(M[~np.eye(M.shape[0],dtype=bool)])

    def record_angle_overlap(self):
        x_vec = xp.sign(xp.sin(self.curr_state[:, :, :, 1]))*xp.cos(self.curr_state[:, :, :, 2])
        y_vec = xp.sign(xp.sin(self.curr_state[:, :, :, 1]))*xp.sin(self.curr_state[:, :, :, 2])
        qxx = xp.einsum("tai,tbi->tab", x_vec, x_vec)/self.Nspins
        qyy = xp.einsum("tai,tbi->tab", y_vec, y_vec)/self.Nspins
        Q = qxx + qyy
        R = qxx - qyy

        for i in range(self.temps.shape[0]):
            Qflat = self.f_rd(Q[i])
            Rflat = self.f_rd(R[i])
            add_to_hist,_,_ = xp.histogram2d(Qflat, Rflat, bins=[79.,79.], range=[[-1.,1.],[-1.,1.]])
            self.angle_overlap_hist[i] += add_to_hist


    def record_radial_hist(self):
        r_vec = self.curr_state[:, :, :, 0]
        for i in range(self.temps.shape[0]):
            add_to_hist,_ = xp.histogram(r_vec[i], bins=50, range=[0.,1.])
            self.radial_hist[i] += add_to_hist

    def record_rho_hist(self):
        r_vec = self.curr_state[:, :, :, 0]*xp.abs(xp.sin(self.curr_state[:, :, :, 1]))
        for i in range(self.temps.shape[0]):
            add_to_hist,_ = xp.histogram(r_vec[i], bins=50, range=[0.,1.])
            self.rho_hist[i] += add_to_hist

    def plot_angle_overlap_distribution(self, tempIdx, ax=None):
        if ax is None:
            fig, ax = plt.subplots()

        x_vec = xp.sign(xp.sin(self.curr_state[tempIdx, :, :, 1]))*xp.cos(self.curr_state[tempIdx, :, :, 2])
        y_vec = xp.sign(xp.sin(self.curr_state[tempIdx, :, :, 1]))*xp.sin(self.curr_state[tempIdx, :, :, 2])
        qxx = xp.einsum("ai,bi->ab", x_vec, x_vec)/self.Nspins
        qyy = xp.einsum("ai,bi->ab", y_vec, y_vec)/self.Nspins

        if GPUAcc:
            ax.hist2d(qxx.flatten().get(), qyy.flatten().get(), bins=[
                      100, 100], range=[[-1.0, 1.0], [-1.0, 1.0]])
        else:
            ax.hist2d(qxx.flatten(), qyy.flatten(), bins=[
                      40, 40], range=[[-1.0, 1.0], [-1.0, 1.0]])
        plt.plot([0.0, 1.0], [1.0, 0.0], 'r-')
        plt.plot([0.0, 1.0], [-1.0, 0.0], 'r-')
        plt.plot([0.0, -1.0], [-1.0, 0.0], 'r-')
        plt.plot([0.0, -1.0], [1.0, 0.0], 'r-')
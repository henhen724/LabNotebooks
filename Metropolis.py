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


def check_CuPy(arr):
    if GPUAcc and isinstance(arr, np.ndarray):
        return xp.array(arr)
    return arr


class Metropolis:
    def __init__(self, Jnonlocal, Jlocal, Zenergy, Tfinal, Nrepl=100, Nspins=20, steps=10000, sigma=0.01, confine_energy_mag=1e12, AnnealT=100.):
        self.Jnonlocal = check_CuPy(Jnonlocal)
        self.Jlocal = check_CuPy(Jlocal)
        self.Zenergy = Zenergy
        self.Tfinal = Tfinal
        self.Nrepl = Nrepl
        self.Nspins = Nspins
        self.steps = steps
        self.sigma = sigma
        self.confine_energy_mag = confine_energy_mag
        self.AnnealT = AnnealT
        self.energy_record = xp.zeros((self.Nrepl, self.steps))
        self.curr_state = xp.zeros((self.Nrepl, self.Nspins, 3))

    def run(self):
        temp = xp.exp(-xp.arange(self.steps)/(10*self.Nspins)) + self.Tfinal

        # indexing for the states
        self.curr_state[:, :, 0] = xp.random.rand(self.Nrepl, self.Nspins)
        self.curr_state[:, :, 1] = xp.pi * \
            xp.random.rand(self.Nrepl, self.Nspins)
        self.curr_state[:, :, 2] = 2*xp.pi * \
            xp.random.rand(self.Nrepl, self.Nspins)

        displacement = xp.zeros((self.Nrepl, self.Nspins, 3))

        for i in range(self.steps-1):
            # print("Step: ", i)
            self.energy_record[:, i] = self.motional_model(self.curr_state)
            # angular displacements
            if xp.random.rand() < 0.5:
                displacement[:, :, 0] = xp.zeros((self.Nrepl, self.Nspins))
                displacement[:, :, 1:] = xp.random.normal(
                    loc=0.0, scale=self.sigma, size=(self.Nrepl, self.Nspins, 2))
            else:
                displacement[:, :, 0] = xp.random.normal(
                    loc=0.0, scale=self.sigma, size=(self.Nrepl, self.Nspins))
                displacement[:, :, 1:] = xp.zeros((self.Nrepl, self.Nspins, 2))

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
                (DeltaE <= 0), (xp.exp(-xp.abs(DeltaE)/temp[i]) > xp.random.rand(1)))
            # print("Decisions: ", state_decissions[0])
            # print(tentative_state.shape, self.curr_state.shape)
            self.curr_state = xp.einsum("i,ijk->ijk", state_decissions, tentative_state) + xp.einsum(
                "i,ijk->ijk", xp.logical_not(state_decissions), self.curr_state)

        self.energy_record[:, self.steps -
                           1] = self.motional_model(self.curr_state)

        return self.curr_state

    def motional_model(self, this_state):
        energy = xp.zeros(self.Nrepl)
        energy += -self.Zenergy * \
            xp.sum(this_state[:, :, 0]*xp.cos(this_state[:, :, 1]), axis=1)
        if xp.isnan(energy).any():
            print("Zenergy threw the nan")
        x_vec = this_state[:, :, 0] * \
            xp.sin(this_state[:, :, 1])*xp.cos(this_state[:, :, 2])
        y_vec = this_state[:, :, 0] * \
            xp.sin(this_state[:, :, 1])*xp.sin(this_state[:, :, 2])
        
        rho = xp.square(x_vec) + xp.square(y_vec)

        energy += -xp.einsum("i,ji->j", self.Jlocal, rho)
        if xp.isnan(energy).any():
            print("Jlocal threw the nan")
        energy += -xp.einsum("fj,jk,fk->f", x_vec, self.Jnonlocal, x_vec) + \
            xp.einsum("fj,jk,fk->f", y_vec, self.Jnonlocal, y_vec)
        if xp.isnan(energy).any():
            print("self.Jnonlocal threw the nan")
        if xp.isnan(xp.einsum("fj,jk,fk->f", x_vec, self.Jnonlocal, x_vec)).any():
            print("X is: ", x_vec, "\n\n R is:",
                  this_state[:, :, 0], "\n\n theta is:", this_state[:, :, 1], "\n\n phi is:", this_state[:, :, 2])
        return energy

    def confine_energy(self, this_state):
        energy = xp.zeros(self.Nrepl)
        # print("State: ", state)
        for i in range(self.Nspins):
            spin = this_state[:, i, :]
            energy += self.confine_energy_mag * \
                xp.abs(np.logical_or(spin[:, 0] >= 1, spin[:,0] < 0))*np.abs(spin[:, 0])
        return energy

    def plot_energy_record(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(np.arange(self.steps), self.energy_record.get().T)

    def plot_single_spin_hist(self, spin, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        x_vec = self.curr_state[:, :, 0] * \
            xp.sin(self.curr_state[:, :, 1]) * \
            xp.cos(self.curr_state[:, :, 2])
        y_vec = self.curr_state[:, :, 0] * \
            xp.sin(self.curr_state[:, :, 1]) * \
            xp.sin(self.curr_state[:, :, 2])
        if GPUAcc:
            ax.hist2d(x_vec[:, spin].get(),
                      y_vec[:, spin].get(), bins=100)
        else:
            ax.hist2d(x_vec[:, spin], y_vec[:, spin], bins=100)

    def plot_overlap_distribution(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()

        x_vec = self.curr_state[:, :, 0] * \
            xp.sin(self.curr_state[:, :, 1]) * \
            xp.cos(self.curr_state[:, :, 2])
        y_vec = self.curr_state[:, :, 0] * \
            xp.sin(self.curr_state[:, :, 1]) * \
            xp.sin(self.curr_state[:, :, 2])
        qxx = xp.einsum("ai,bi->ab", x_vec[:], x_vec[:])/self.Nspins
        qyy = xp.einsum("ai,bi->ab", y_vec[:], y_vec[:])/self.Nspins

        if GPUAcc:
            ax.hist2d(qxx.flatten().get(), qyy.flatten().get(), bins=[
                      40, 40], range=[[-1.0, 1.0], [-1.0, 1.0]])
        else:
            ax.hist2d(qxx.flatten(), qyy.flatten(), bins=[
                      40, 40], range=[[-1.0, 1.0], [-1.0, 1.0]])
        plt.plot([0.0, 1.0], [1.0, 0.0], 'r-')
        plt.plot([0.0, 1.0], [-1.0, 0.0], 'r-')
        plt.plot([0.0, -1.0], [-1.0, 0.0], 'r-')
        plt.plot([0.0, -1.0], [1.0, 0.0], 'r-')


class Metropolis_Time_Recorded:
    def __init__(self, Jnonlocal, Jlocal, Zenergy, Tfinal, Nrepl=100, Nspins=20, steps=1000, sigma=0.01, confine_energy_mag=1e33):
        self.Jnonlocal = Jnonlocal
        self.Jlocal = Jlocal
        self.Zenergy = Zenergy
        self.Tfinal = Tfinal
        self.Nrepl = Nrepl
        self.Nspins = Nspins
        self.steps = steps
        self.sigma = sigma
        self.confine_energy_mag = confine_energy_mag

    def run(self):
        temp = xp.exp(-xp.arange(self.steps)/(10*self.Nspins)) + self.Tfinal

        # indexing for the states
        self.states = xp.zeros((self.Nrepl, self.steps, self.Nspins, 3))
        self.states[:, 0, :, 0] = xp.random.rand(self.Nrepl, self.Nspins)
        self.states[:, 0, :, 1] = xp.pi * \
            xp.random.rand(self.Nrepl, self.Nspins)
        self.states[:, 0, :, 2] = 2*xp.pi * \
            xp.random.rand(self.Nrepl, self.Nspins)

        displacement = xp.zeros((self.Nrepl, self.Nspins, 3))

        for i in range(self.steps-1):
            # print("Step: ", i)
            # angular displacements
            if xp.random.rand() < 0.5:
                displacement[:, :, 0] = xp.zeros((self.Nrepl, self.Nspins))
                displacement[:, :, 1:] = xp.random.normal(
                    loc=0.0, scale=self.sigma, size=(self.Nrepl, self.Nspins, 2))
            else:
                displacement[:, :, 0] = xp.random.normal(
                    loc=0.0, scale=self.sigma, size=(self.Nrepl, self.Nspins))
                displacement[:, :, 1:] = xp.zeros((self.Nrepl, self.Nspins, 2))

            tentative_state = displacement + self.states[:, i]

            # print("Enegies NAN: ")
            # print(xp.isnan(confine_energy(tentative_state)))
            # print(xp.isnan(confine_energy(self.states[:,i])))
            # print(xp.isnan(motional_model(tentative_state, Zenergy, Jlocal, self.Jnonlocal)))
            # print(xp.isnan(motional_model(self.states[:,i], Zenergy, Jlocal, self.Jnonlocal)))
            # print("\n")

            DeltaE = self.confine_energy(tentative_state) - self.confine_energy(self.states[:, i]) + self.motional_model(
                tentative_state) - self.motional_model(self.states[:, i])

            # print("DE: ", DeltaE[0], "R before", self.states[0,i,:,0], "R after", tentative_state[0,:,0])

            state_decissions = xp.logical_or(
                (DeltaE <= 0), (xp.exp(-xp.abs(DeltaE)/temp[i]) > xp.random.rand(1)))
            # print("Decisions: ", state_decissions[0])
            # print(tentative_state.shape, self.states.shape)
            self.states[:, i+1] = xp.einsum("i,ijk->ijk", state_decissions, tentative_state) + xp.einsum(
                "i,ijk->ijk", xp.logical_not(state_decissions), self.states[:, i])
        return self.states

    def motional_model(self, this_state):
        energy = xp.zeros(self.Nrepl)
        energy += -self.Zenergy * \
            xp.sum(this_state[:, :, 0]*xp.cos(this_state[:, :, 1]), axis=1)
        if xp.isnan(energy).any():
            print("Zenergy threw the nan")
        energy += -xp.einsum("i,ji->j", self.Jlocal,
                             xp.square(this_state[:, :, 0]*xp.sin(this_state[:, :, 1])))
        if xp.isnan(energy).any():
            print("Jlocal threw the nan")
        x_vec = this_state[:, :, 0] * \
            xp.sin(this_state[:, :, 1])*xp.cos(this_state[:, :, 2])
        y_vec = this_state[:, :, 0] * \
            xp.sin(this_state[:, :, 1])*xp.sin(this_state[:, :, 2])
        energy += -xp.einsum("fj,jk,fk->f", x_vec, self.Jnonlocal, x_vec) + \
            xp.einsum("fj,jk,fk->f", y_vec, self.Jnonlocal, y_vec)
        if xp.isnan(energy).any():
            print("self.Jnonlocal threw the nan")
        if xp.isnan(xp.einsum("fj,jk,fk->f", x_vec, self.Jnonlocal, x_vec)).any():
            print("X is: ", x_vec, "\n\n R is:",
                  this_state[:, :, 0], "\n\n theta is:", this_state[:, :, 1], "\n\n phi is:", this_state[:, :, 2])
        return energy

    def confine_energy(self, this_state):
        energy = xp.zeros(self.Nrepl)
        # print("State: ", state)
        for i in range(self.Nspins):
            spin = this_state[:, i, :]
            energy += self.confine_energy_mag * \
                (xp.abs(spin[:, 0] >= 1))*spin[:, 0]
        return energy

    def plot_single_spin_hist(self, spin, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        x_vec = self.states[:, :, :, 0] * \
            xp.sin(self.states[:, :, :, 1])*xp.cos(self.states[:, :, :, 2])
        y_vec = self.states[:, :, :, 0] * \
            xp.sin(self.states[:, :, :, 1])*xp.sin(self.states[:, :, :, 2])
        if GPUAcc:
            ax.hist2d(x_vec[1, :, spin].get(),
                      y_vec[1, :, spin].get(), bins=100)
        else:
            ax.hist2d(x_vec[1, :, spin], y_vec[1, :, spin], bins=100)

    def plot_overlap_distribution(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()

        x_vec = self.states[:, :, :, 0] * \
            xp.sin(self.states[:, :, :, 1])*xp.cos(self.states[:, :, :, 2])
        y_vec = self.states[:, :, :, 0] * \
            xp.sin(self.states[:, :, :, 1])*xp.sin(self.states[:, :, :, 2])
        qxx = xp.einsum("ai,bi->ab", x_vec[:, 999], x_vec[:, 999])/self.Nspins
        qyy = xp.einsum("ai,bi->ab", y_vec[:, 999], y_vec[:, 999])/self.Nspins

        if GPUAcc:
            ax.hist2d(qxx.flatten().get(), qyy.flatten().get(), bins=[
                      40, 40], range=[[-1.0, 1.0], [-1.0, 1.0]])
        else:
            ax.hist2d(qxx.flatten(), qyy.flatten(), bins=[
                      40, 40], range=[[-1.0, 1.0], [-1.0, 1.0]])
        plt.plot([0.0, 1.0], [1.0, 0.0], 'r-')
        plt.plot([0.0, 1.0], [-1.0, 0.0], 'r-')
        plt.plot([0.0, -1.0], [-1.0, 0.0], 'r-')
        plt.plot([0.0, -1.0], [1.0, 0.0], 'r-')



class Metropolis2D:
    def __init__(self, Jnonlocal, Tfinal, Nrepl=100, Nspins=8, steps=10000, sigma=xp.pi/10, confine_energy_mag=1e12, AnnealT=100, AnnealHigh=1.5):
        self.Jnonlocal = check_CuPy(Jnonlocal)
        self.Tfinal = Tfinal
        self.Nrepl = Nrepl
        self.Nspins = Nspins
        self.steps = steps
        self.sigma = sigma
        self.confine_energy_mag = confine_energy_mag
        self.AnnealT = AnnealT
        self.AnnealHigh = AnnealHigh
        self.energy_record = xp.zeros((self.Nrepl, self.steps))
        self.curr_state = xp.zeros((self.Nrepl, self.Nspins))

    def run(self):
        temp = self.AnnealHigh*xp.exp(-xp.arange(self.steps)/self.AnnealT) + self.Tfinal

        # indexing for the states
        self.curr_state = 2*xp.pi * \
            xp.random.rand(self.Nrepl, self.Nspins)

        displacement = xp.zeros((self.Nrepl, self.Nspins))

        for i in range(self.steps-1):
            # print("Step: ", i)
            self.energy_record[:, i] = self.motional_model(self.curr_state)
            # angular displacements
            displacement = xp.random.normal(
                loc=0.0, scale=self.sigma, size=(self.Nrepl, self.Nspins))

            tentative_state = displacement + self.curr_state

            # print("Enegies NAN: ")
            # print(xp.isnan(confine_energy(tentative_state)))
            # print(xp.isnan(confine_energy(self.curr_state[:,i])))
            # print(xp.isnan(motional_model(tentative_state, Zenergy, Jlocal, self.Jnonlocal)))
            # print(xp.isnan(motional_model(self.curr_state[:,i], Zenergy, Jlocal, self.Jnonlocal)))
            # print("\n")

            DeltaE = self.motional_model(
                tentative_state) - self.motional_model(self.curr_state)

            # print("DE: ", DeltaE[0], "R before", self.curr_state[0,:,0], "R after", tentative_state[0,:,0])

            state_decissions = xp.logical_or(
                (DeltaE <= 0), (xp.exp(-xp.abs(DeltaE)/temp[i]) > xp.random.rand(1)))
            # print("Decisions: ", state_decissions[0])
            # print(tentative_state.shape, self.curr_state.shape)
            self.curr_state = xp.einsum("i,ij->ij", state_decissions, tentative_state) + xp.einsum(
                "i,ij->ij", xp.logical_not(state_decissions), self.curr_state)

        self.energy_record[:, self.steps -
                           1] = self.motional_model(self.curr_state)

        return self.curr_state

    def motional_model(self, this_state):
        energy = xp.zeros(self.Nrepl)
        x_vec = xp.cos(this_state)
        y_vec = xp.sin(this_state)
        energy += -xp.einsum("fj,jk,fk->f", x_vec, self.Jnonlocal, x_vec) + \
            xp.einsum("fj,jk,fk->f", y_vec, self.Jnonlocal, y_vec)
        if xp.isnan(energy).any():
            print("self.Jnonlocal threw the nan")
        if xp.isnan(xp.einsum("fj,jk,fk->f", x_vec, self.Jnonlocal, x_vec)).any():
            print("X is: ", x_vec, "\n\n R is:", "\n\n phi is:", this_state)
        return energy

    def plot_energy_record(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(np.arange(self.steps), self.energy_record.get().T)

    def plot_single_spin_hist(self, spin, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        x_vec = xp.cos(self.curr_state)
        y_vec = xp.sin(self.curr_state)
        if GPUAcc:
            ax.hist2d(x_vec[:, spin].get(),
                      y_vec[:, spin].get(), bins=100)
        else:
            ax.hist2d(x_vec[:, spin], y_vec[:, spin], bins=100)

    def plot_overlap_distribution(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()

        x_vec = xp.cos(self.curr_state)
        y_vec = xp.sin(self.curr_state)
        qxx = xp.einsum("ai,bi->ab", x_vec, x_vec)/self.Nspins
        qyy = xp.einsum("ai,bi->ab", y_vec, y_vec)/self.Nspins

        if GPUAcc:
            ax.hist2d(qxx.flatten().get(), qyy.flatten().get(), bins=[
                      40, 40], range=[[-1.0, 1.0], [-1.0, 1.0]])
        else:
            ax.hist2d(qxx.flatten(), qyy.flatten(), bins=[
                      40, 40], range=[[-1.0, 1.0], [-1.0, 1.0]])
        plt.plot([0.0, 1.0], [1.0, 0.0], 'r-')
        plt.plot([0.0, 1.0], [-1.0, 0.0], 'r-')
        plt.plot([0.0, -1.0], [-1.0, 0.0], 'r-')
        plt.plot([0.0, -1.0], [1.0, 0.0], 'r-')
import pickle
from ParrellelTempering import ParrallelTempering
import matplotlib.pyplot as plt
import os
GPUAcc = False
try:
    import cupy as xp
    GPUAcc = True
    print("Using GPU Acceleration")
    import numpy as np
except:
    import numpy as xp


def run():
    assert(GPUAcc)
    mempool = xp.get_default_memory_pool()

    Nspins = 8

    Zenergy = 1.0

    gs = [1.25] #[0.5,1.0,1.25,2.0, 2.25, 2.5]

    Jlocals = xp.load("PosBasedJlocals.npy")
    Jnonlocals = xp.load("PosBasedJnonlocals.npy")
    Ks = xp.load("PosBasedKs.npy")

    for i in range(Jlocals.shape[0]):
        # JK_nonlocal = xp.load("Measured_JK_Matrices/"+file)
        # Jnonlocal = xp.real(JK_nonlocal)
        # K = xp.imag(JK_nonlocal)
        Jnonlocal = Jnonlocals[i]
        K = Ks[i]
        Jlocal = Jlocals[i]

        M = np.block([[xp.asnumpy(Jnonlocal), xp.asnumpy(K)],[xp.asnumpy(K), -xp.asnumpy(Jnonlocal)]])
        M = xp.array(M)

        # Jlocal = 10.*xp.mean(Jnonlocal)*xp.ones(8)

        Tc = max(xp.linalg.eigvalsh(M))

        Mprime = np.block([[xp.asnumpy(Jnonlocal + xp.diag(Jlocal)), xp.asnumpy(K)],[xp.asnumpy(K), -xp.asnumpy(Jnonlocal + xp.diag(Jlocal))]])
        Mprime = xp.array(Mprime)

        N = max(xp.linalg.eigvalsh(Mprime))

        Jnonlocal = Jnonlocal/N
        K = K/N
        Jlocal = Jlocal/N

        #[2.25, 2.5, 3.0, 5.0]

        temps = xp.logspace(xp.log(0.001)/xp.log(10.),xp.log(1.5)/xp.log(10.), 60)
        for g in gs:
            print(f"Pre-Metropolis GPU Memory {mempool.used_bytes()/2**20:.2f} Mb")
            met = ParrallelTempering(g*Jnonlocal, g*Jlocal, g*K, Zenergy, temps*Tc,
                                steps=10000, sigma=xp.pi/30, sigmaR=0.05, Nrepl=1000, Nspins=Nspins)
            final_state = met.run()
            print(f"Metropolis GPU Memory {mempool.used_bytes()/2**20:.2f} Mb")
            met.curr_state = None
            met.energy_record = None
            del met.curr_state
            del met.energy_record
            with open(f"MetropolisRuns/FinalVersion/PTpos{i}g={g:.2f}.pickle", "wb") as f:
                pickle.dump(met, f)
            del met


if __name__ == "__main__":
    run()

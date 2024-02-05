import pickle
from MetropolisV2 import Metropolis
import matplotlib.pyplot as plt
GPUAcc = False
try:
    import cupy as xp
    GPUAcc = True
    print("Using GPU Acceleration")
except:
    import numpy as xp


def run():
    assert(GPUAcc)
    mempool = xp.get_default_memory_pool()

    Nspins = 8

    Zenergy = 1.0

    Jlocals = xp.load("ConfocalJlocals_v2.npy")
    Ks = xp.load("ConfocalKs_v2.npy")
    Jnonlocals = xp.load("ConfocalJnonlocals_v2.npy")

    gs = [0.5,1.0,1.25,2.0, 2.25, 2.5]#[2.25, 2.5, 3.0, 5.0]

    for i in range(Jlocals.shape[0]):
        Tc = max(xp.abs(xp.linalg.eigvalsh(Jnonlocals[i])))

        Tfinal = 0.0001*Tc
        AnnealHigh = Tfinal*2.0
        AnnealT = 5000
        for g in gs:
            print(f"Pre-Metropolis GPU Memory {mempool.used_bytes()/2**20:.2f} Mb")
            met = ParrallelTempering(2.5*Jnonlocals[8], 2.5*Jlocals[8], 2.5*Ks[8], Zenergy, temps,
                                steps=1000, sigma=xp.pi/40., Nrepl=500, Nspins=Nspins)
            final_state = met.run()
            print(f"Metropolis GPU Memory {mempool.used_bytes()/2**20:.2f} Mb")
            with open(f"MetropolisRuns/3component/J{i}g={g:.2f}T=0.0001Tc.pickle", "wb") as f:
                pickle.dump(met, f)
            del met


if __name__ == "__main__":
    run()

import pickle
from Metropolis import Metropolis_Time_Recorded, Metropolis, Metropolis2D
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
    Nrepl = 500

    Jlocals = xp.load("ConfocalJlocals.npy")
    Jnonlocals = xp.load("ConfocalJnonlocals.npy")

    gs = [1.50, 1.75, 2.50, 3.0]

    for i in range(Jlocals.shape[0]):
        Tc = max(xp.abs(xp.linalg.eigvalsh(Jnonlocals[i])))

        Tfinal = 0.05*Tc
        AnnealHigh = Tfinal*1.5
        AnnealT = 5000
        for g in gs:
            print(f"Pre-Metropolis GPU Memory {mempool.used_bytes()/2**20:.2f} Mb")
            met = Metropolis2D(g*Jnonlocals[i], Tfinal, steps=int(3*AnnealT), sigma=xp.pi/10, Nspins=Nspins, AnnealT=AnnealT, Nrepl=1000, AnnealHigh=AnnealHigh)
            final_state = met.run()
            print(f"Metropolis GPU Memory {mempool.used_bytes()/2**20:.2f} Mb")
            with open(f"MetropolisRuns/2component/J{i}g={g:.2f}T=0.05Tc.pickle", "wb") as f:
                pickle.dump(met, f)
            del met


if __name__ == "__main__":
    run()

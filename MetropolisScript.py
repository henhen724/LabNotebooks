import pickle
from Metropolis import Metropolis_Time_Recorded, Metropolis
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

    Nspins = 20

    Jnonlocal = 2*xp.random.rand(Nspins, Nspins)-1
    Jlocal = 10.*xp.ones(Nspins)

    Zenergies = xp.logspace(-5.0, 2.0, 8)
    Tfinals = xp.logspace(-8.0, 2.0, 11)
    for Tfinal in Tfinals:
        for Zenergy in Zenergies:
            print(f"Pre-Metropolis GPU Memory {mempool.used_bytes()/2**20:.2f} Mb")
            met = Metropolis(Jnonlocal, Jlocal, Zenergy, Tfinal, steps=20000)
            final_state = met.run()
            print(f"Metropolis GPU Memory {mempool.used_bytes()/2**20:.2f} Mb")
            with open(f"MetropolisRuns/Ze={Zenergy:.1e},T={Tfinal:.1e}.pickle", "wb") as f:
                pickle.dump(met, f)
            del met


if __name__ == "__main__":
    run()

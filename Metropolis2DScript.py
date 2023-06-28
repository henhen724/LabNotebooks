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
    with open("MetropolisRuns/3component/Ze=1.0e+00,T=1.0e+00.pickle", "rb") as f:
        loadedmet = pickle.load(f)

    assert(GPUAcc)
    mempool = xp.get_default_memory_pool()

    Nspins = loadedmet.Nspins

    Jnonlocal = loadedmet.Jnonlocal
    Jlocal = loadedmet.Jlocal

    Tfinals = xp.logspace(-8.0, 2.0, 11)
    for Tfinal in Tfinals:
        print(f"Pre-Metropolis GPU Memory {mempool.used_bytes()/2**20:.2f} Mb")
        met = Metropolis2D(Jnonlocal, Tfinal, steps=20000, sigma=0.05, Nspins=Nspins)
        final_state = met.run()
        print(f"Metropolis GPU Memory {mempool.used_bytes()/2**20:.2f} Mb")
        with open(f"MetropolisRuns/2component/T={Tfinal:.1e}.pickle", "wb") as f:
            pickle.dump(met, f)
        del met


if __name__ == "__main__":
    run()

import pickle
from ParrellelTempering import ParrallelTempering
import matplotlib.pyplot as plt
import os
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

    gs = [0.5,1.0,1.25,2.0, 2.25, 2.5]

    for file in os.listdir("Measured_JK_Matrices"):
        JK_nonlocal = xp.load("Measured_JK_Matrices/"+file)
        Jlocal = 10.*xp.mean(xp.abs(JK_nonlocal))*xp.ones(8)
        Jnonlocal = xp.real(JK_nonlocal)
        K = xp.imag(JK_nonlocal)
        
        Tc = max(xp.abs(xp.linalg.eigvalsh(Jnonlocal)))

        N = 2*max(Jlocal)+2*Tc

        Jnonlocal = Jnonlocal/N
        K = K/N

        #[2.25, 2.5, 3.0, 5.0]

        temps = xp.linspace(0.01,1.5, 70)
        for g in gs:
            print(f"Pre-Metropolis GPU Memory {mempool.used_bytes()/2**20:.2f} Mb")
            met = ParrallelTempering(g*Jnonlocal, g*Jlocal, g*K, Zenergy, temps*Tc,
                                steps=10000, sigma=xp.pi/20, sigmaR=0.05, Nrepl=1000, Nspins=Nspins)
            final_state = met.run()
            print(f"Metropolis GPU Memory {mempool.used_bytes()/2**20:.2f} Mb")
            with open(f"MetropolisRuns/3component/PT{file}g={g:.2f}.pickle", "wb") as f:
                pickle.dump(met, f)
            del met


if __name__ == "__main__":
    run()

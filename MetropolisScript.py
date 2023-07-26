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

    Nspins = 8

    Jnonlocal = xp.array([[-1.46000754,  0.66307027, -0.36324195, -0.89853853,  0.10771147,
                           -0.36567651, -0.58853263,  0.16587756],
                          [0.66307027, -0.96435315, -0.01312276, -0.79007184,  0.30340302,
                           0.38625942,  1.19862103,  0.15204514],
                          [-0.36324195, -0.01312276, -0.11171844,  1.06908322, -1.24787081,
                           -0.63673043,  0.74883775,  0.35162467],
                          [-0.89853853, -0.79007184,  1.06908322, -1.66207899, -0.35355957,
                           -0.54518182,  0.36818289,  1.05812929],
                          [0.10771147,  0.30340302, -1.24787081, -0.35355957, -1.5087697,
                           0.17371169, -0.08017211,  1.51076996],
                          [-0.36567651,  0.38625942, -0.63673043, -0.54518182,  0.17371169,
                           1.72453836,  0.01690911,  0.35817871],
                          [-0.58853263,  1.19862103,  0.74883775,  0.36818289, -0.08017211,
                           0.01690911, -0.4166188, -0.52707744],
                          [0.16587756,  0.15204514,  0.35162467,  1.05812929,  1.51076996,
                           0.35817871, -0.52707744, -0.78813721]])
    Jlocal = 10.*xp.ones(Nspins)

    Zenergies = xp.linspace(24.0, 28.0, 3)
    Tc=max(xp.linalg.eigvalsh(Jnonlocal))
    Tfinal =  0.05*Tc

    AnnealHigh = max(xp.linalg.eigvalsh(Jnonlocal))*1.5
    AnnealT = 50000
    Nrepl=500
    for Zenergy in Zenergies:
        print(
            f"Pre-Metropolis GPU Memory {mempool.used_bytes()/2**20:.2f} Mb")
        met = Metropolis(Jnonlocal, Jlocal, Zenergy, Tfinal,
                            steps=int(3*AnnealT), sigma=xp.pi/10., Nrepl=Nrepl, Nspins=Nspins, AnnealT=AnnealT, AnnealHigh=AnnealHigh)
        final_state = met.run()
        print(f"Metropolis GPU Memory {mempool.used_bytes()/2**20:.2f} Mb")
        with open(f"MetropolisRuns/3component/Ze={Zenergy:.2e},T={Tfinal/Tc:.1e}.pickle", "wb") as f:
            pickle.dump(met, f)
        del met


if __name__ == "__main__":
    run()

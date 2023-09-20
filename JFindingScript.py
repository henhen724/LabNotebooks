import pickle
from MetropolisV2 import Metropolis, Metropolis2D
import matplotlib.pyplot as plt
GPUAcc = False
try:
    import cupy as xp
    GPUAcc = True
    print("Using GPU Acceleration")
except:
    import numpy as xp


def harm_greens_fnc_v2(posx, posy, sigma, tval):
    N = posx.shape[0]
    g = (1-2*sigma**2) / (1+2*sigma**2) # gamma factor

    normalization = (1+g)**2/(4*(1-g**2*tval**2)) # Differs by the paper normalization by a missing 1/4
                                                  # This comes from Brandan's code base and make point particles have t=1.
    x = xp.einsum("i,j->ij",posx,xp.ones(N)) 
    xT = xp.einsum("i,j->ij",xp.ones(N),posx)

    y = xp.einsum("i,j->ij",posy,xp.ones(N)) 
    yT = xp.einsum("i,j->ij",xp.ones(N),posy)

    sumOfSquares = 0.5*(xp.square(x - xT) + xp.square(y - yT) + xp.square(x + xT) + xp.square(y + yT))
    dotProduct = 0.25*(xp.square(x + xT) + xp.square(y + yT) - xp.square(x - xT) - xp.square(y - yT))

    term1 = (1+g*tval**2)*sumOfSquares
    term2 = -2*(1+g)*tval*dotProduct

    return normalization*xp.exp(-(1+g)/(4*(1-g**2*tval**2))*2*(term1 + term2))

def K_fnc(posx, posy, sigma):
    rdot = (xp.outer(posx,posx) + xp.outer(posy,posy))
    return 8 * sigma**2 * rdot * xp.sin(2*rdot)

def harm_greens_fnc_diagonal_v2(posx, posy, sigma, tval):
    N = posx.shape[0]
    g = (1-2*sigma**2) / (1+2*sigma**2) # gamma factor

    normalization = (1+g)**2/(4*(1-g**2*tval**2)) # Differs by the paper normalization by a missing 1/4 from Brandan

    term1 = (1+g*tval**2)*2*(xp.square(posx) + xp.square(posy))
    term2 = -2*(1+g)*tval*(xp.square(posx) + xp.square(posy))

    return normalization*xp.exp(-(1+g)/(4*(1-g**2*tval**2))*2*(term1 + term2))

def confoncal_greens_fnc_nonlocal_v2(posx, posy, sigma, tval):
    return 2*xp.real(harm_greens_fnc_v2(posx, posy, sigma, 1j*tval))

def confoncal_greens_fnc_local_v2(posx, posy, sigma, tval):
    return harm_greens_fnc_diagonal_v2(posx, posy, sigma, tval) + harm_greens_fnc_diagonal_v2(posx, posy, sigma, -tval)

def generate_Js_v2(pos_sigma = 0.2, sigma_A=4.0/35.2):
    posx = xp.random.normal(loc=0,scale=pos_sigma, size=Nspins)
    posy = xp.random.normal(loc=0,scale=pos_sigma, size=Nspins)

    Jnonlocal = xp.real(confoncal_greens_fnc_nonlocal_v2(posx,posy, sigma_A,1.0))
    K = K_fnc(posx, posy, sigma_A)
    Jlocal = xp.real(confoncal_greens_fnc_local_v2(posx, posy, sigma_A,1.0))
    return Jnonlocal, K, Jlocal

def run(i):
    assert(GPUAcc)
    mempool = xp.get_default_memory_pool()

    Nspins = 8
    Nrepl = 500

    pos_sigma=2.0
    posx = xp.random.normal(loc=0,scale=pos_sigma, size=Nspins)
    posy = xp.random.normal(loc=0,scale=pos_sigma, size=Nspins)

    Jnonlocal = xp.real(confoncal_greens_fnc_nonlocal_v2(posx,posy, 4.0/35.2,1.0))
    K = K_fnc(posx, posy, 4.0/35.2)
    Jlocal = xp.real(confoncal_greens_fnc_local_v2(posx, posy, 4.0/35.2,1.0))

    Tc = max(xp.abs(xp.linalg.eigvalsh(Jnonlocal)))

    N = 2*max(Jlocal)+2*Tc

    Jnonlocal = Jnonlocal/N
    K = K/N
    Jlocal = Jlocal/N

    Tfinal = 0.001*Tc
    AnnealHigh = Tfinal*1.5
    AnnealT = 5000

    print(f"Pre-Metropolis GPU Memory {mempool.used_bytes()/2**20:.2f} Mb")
    met = Metropolis2D(Jnonlocal, K, Tfinal, steps=int(3*AnnealT), sigma=xp.pi/10., Nrepl=Nrepl, Nspins=Nspins, AnnealT=AnnealT, AnnealHigh=AnnealHigh, Jlocal=Jlocal)
    final_state = met.run()
    print(f"Metropolis GPU Memory {mempool.used_bytes()/2**20:.2f} Mb")
    with open(f"MetropolisRuns/FindingJs/{i}.pickle", "wb") as f:
        pickle.dump(met, f)
    del met


if __name__ == "__main__":
    for i in range(150):
        run(i)

import pickle
from MetropolisV2 import Metropolis, Metropolis2D
import matplotlib.pyplot as plt
GPUAcc = False
try:
    import cupy as xp
    GPUAcc = True
    print("Using GPU Acceleration")
    import numpy as np
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
    N = posx.shape[0]
    weff = xp.sqrt(1 + 4*sigma**4)

    x = xp.einsum("i,j->ij",posx,xp.ones(N)) 
    xT = xp.einsum("i,j->ij",xp.ones(N),posx)

    y = xp.einsum("i,j->ij",posy,xp.ones(N)) 
    yT = xp.einsum("i,j->ij",xp.ones(N),posy)

    rdot = 0.25*(xp.square(x + xT) + xp.square(y + yT) - xp.square(x - xT) - xp.square(y - yT)) / weff**2 # calculates ri dot rj / w0^2
    rnorm = 0.5*(xp.square(x - xT) + xp.square(y - yT) + xp.square(x + xT) + xp.square(y + yT)) # calculates (ri^2 + rj^2) / w0^2
    
    prefactor = - 2*sigma**2 / weff**2
    expenvelope = xp.exp(-2*sigma**2*rnorm/weff**2)
    
    # return 8 * sigmaA**2 / w0**2 * rdot * xp.sin(2*rdot)
    return prefactor * expenvelope * (4*rdot * xp.sin(2*rdot) - 2*prefactor * rnorm * xp.cos(2*rdot))
    # rdot = (xp.outer(posx,posx) + xp.outer(posy,posy))
    # return 8 * sigma**2 * rdot * xp.sin(2*rdot)

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

def generate_Js_v2(pos_sigma = 0.2, Nspins=8, sigma_A=4.0/35.2):
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

    posx = xp.load("J_Matrix_Positions/allX.npy")
    posy = xp.load("J_Matrix_Positions/allY.npy")

    posx = posx/35.2e-6
    posy = posy/35.2e-6

    Jnonlocals = xp.zeros((posx.shape[0], 8, 8))
    Ks = xp.zeros((posx.shape[0], 8, 8))
    Jlocals = xp.zeros((posx.shape[0], 8))

    GoodIndeces = []

    for i in range(posx.shape[0]):
        if (not xp.isnan(posx[i]).any()) and (not xp.isnan(posy[i]).any()):
            GoodIndeces.append(i)

        Jnonlocals[i] = xp.real(confoncal_greens_fnc_nonlocal_v2(posx[i],posy[i], 4.0/35.2,1.0))
        Ks[i] = K_fnc(posx[i], posy[i], 4.0/35.2)
        Jlocals[i] = xp.real(confoncal_greens_fnc_local_v2(posx[i], posy[i], 4.0/35.2,1.0))

        M = np.block([[xp.asnumpy(Jnonlocals[i]), xp.asnumpy(Ks[i])],[xp.asnumpy(Ks[i]), -xp.asnumpy(Jnonlocals[i])]])
        M = xp.array(M)

        Tc = max(xp.linalg.eigvalsh(M))

        N = max(Jlocals[i])+Tc

        Jnonlocals[i] = Jnonlocals[i]/N
        Ks[i] = Ks[i]/N
        Jlocals[i] = Jlocals[i]/N
    print(GoodIndeces)
    xp.save("PosBasedJnonlocals.npy", Jnonlocals[GoodIndeces])
    xp.save("PosBasedKs.npy", Ks[GoodIndeces])
    xp.save("PosBasedJlocals.npy", Jlocals[GoodIndeces])




if __name__ == "__main__":
    run(0)

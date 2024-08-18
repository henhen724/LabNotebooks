---
marp: true
math: mathjax
paginate: true
---
# Mean Field Simulation of the Dicke Model with Feedback

---
Add slide deriving or at least stating full mean field equations for the Dicke model.

---
## Current Feedback is Technically Singular
The equations of motion for a feedback system include singular terms

$$I(t) = \sqrt{2\kappa}\langle a \rangle + \frac{dZ}{dt}$$
$$d \ket{\psi} = -i H[I(t)]\ket{\psi} dt + ...$$

$\frac{dZ}{dt}$ is a complex valued white noise, which means in a typical sample it is infinite at very point in time. The current output of real systems have some response time, and therefore never see this infinity. Of course, this also implies that the SDE integrate does not have a consistent way to interpret I(t).

---
The real current is filtered over some timescale deliberately or by the limitations of the electronics. The filtered output can be represented as the raw current convolved with the __filter function__.
$$O_{filt.}(t) = \int_{0}^{\infty} f(\tau) O(t - \tau) d\tau$$
Let $F(\tau) = \Theta(\tau) f(\tau)$
$$ = \int_{-\infty}^{\infty} F(\tau) O(t - \tau) d\tau = F * O (t)$$
which implies
$$\hat{O}_{filt.}(\omega) = \hat{F}(\omega) \hat{O}(\omega)$$
 
---
Filter functions should satisfy the following properties:
$$\int_{-\infty}^{\infty} F(\tau) d\tau = 1 \leftrightarrow \hat{F}(0) = 1$$
$$|\hat{F}(\omega)| \leq 1$$
... and maybe some more. These are just the obvious ones.

---
We can use charge instead of current to integrate the SDE, which will not have singular equations of motion
$$\frac{dQ}{dt} = I(t)$$
$$dQ = \sqrt{2 \kappa} \langle a \rangle dt + dZ$$
I can then relate Q to the filtered current.
$$I_{filt.}(t) = \int_0^{\infty} f(\tau) I(t - \tau) d\tau = \int_0^{\infty} f(\tau) \frac{dQ}{dt}(t - \tau) d\tau$$
$$= f(0)Q(t) - \int_0^{\infty} f'(\tau) Q(t - \tau) d\tau$$
This allows you to use an SDE integrator on a system with current feedback, but __the equations are now apparently time non-local.__

---
# The exponential filter
The cannonical example of a filtering function is the exponential filter
$$f(\tau) = \frac{e^{-\tau/T_{filt.}}}{T_{filt.}}$$
This is the filtering function will use for the rest of the presentation/project. This filtering function has the advantage that is a rescaling of the retarded Green's function of the simplest differential equation.
$$dx = -\frac{x}{T_{filt.}} dt$$

---
# Exponential filtering as an additional equation of motion
This means that the integral term for the filter current, which I call $I_{mem.}$ satifies the following differential equation.
$$I_{mem.} = - \int_0^{\infty} f'(\tau) Q(t - \tau) d\tau = - \int_0^{\infty} e^{-\tau/T_{filt.}} Q(t - \tau)/T_{filt.}^2 d\tau$$
$$dI_{mem.} = (-\frac{I_{mem.}}{T_{filt}}-\frac{Q}{T_{filt.}^2})dt$$
So, now $I_{filt.}$ can be written:
$$I_{filt.}(t) = Q(t)/T_{filt.} + I_{mem.}(t)$$

___
# Checking this works
```julia
using DifferentialEquations, Plots
QhetConj = 1
IfiltMem = 2

Tfilt1=0.1

u0 = ComplexF32[0., 0.]

Iraw(t) = 10*t

function filt!(du, u, p, t)
    du[QhetConj] = Iraw(t)
    du[IfiltMem] = -u[IfiltMem]/Tfilt1 -u[QhetConj]/(Tfilt1^2)
end

function σ_filt!(du, u, p, t)
    @inbounds du[QhetConj] = 0.0
    @inbounds du[IfiltMem] = 0.0
end

prob_dicke = SDEProblem{true}(filt!, σ_filt!, u0, (0.0, tmax), pinit)
sol = solve(prob_dicke, SRA2(), save_noise=true)
```

---
![bg fit](img/InputVSFilt.png)
![bg fit](img/ChargeAndMem.png)
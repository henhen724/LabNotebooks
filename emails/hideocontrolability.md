Ben here is the email I am planning to send to Hideo. At group meeting, you asked me to send it to you first.


Dear Hideo,

I was reading a book on control theory and came across the conditions for controllability of a linear plant with additive control. You probably are already familiar with these, but to make the analogy that I will make latter clear let me be explicit.
$$\frac{d\vec{x}}{dt} = \hat{A}\vec{x} + \hat{B}\vec{u}$$
Controlability meaning given some $x_{target}$ and T, there exists some choice of $u(t)$ such that $x(T) = x_{target}$. This problem can be quite easily solved since there is a closed form soltion to the equations of motion above:
$$\vec{x}(t) = exp(t\hat{A})\vec{x}(0) + \int_{0}^{t} exp(\tau \hat{A}) \hat{B} \vec{u}(t - \tau) d\tau$$
If $\vec{x}$ is dimension $N$, then $\hat{A}$ is an $N \times N$ matrix and therefore satisfies some characteristic equation of degree $N$. So, the exponential can be written as a degree $N-1$ polynomial in A with time varying coefficents.

$$\vec{x}(t) = exp(t\hat{A})\vec{x}(0) + \sum_{n=0}^{N-1} \hat{A}^{n} \hat{B} \vec{c}_{n}(t)$$

You can then set $c_n(t)$ using $u(t)$. Therefore, you only need to check that sum on the right spans the entire space to prove controllability which is equivalent to whether the following rectangular matrix has rank at least $N$:

$$[\hat{B} \;\; \hat{A} \hat{B} \;\; \hat{A}^2 \hat{B}\;\;...\;\; \hat{A}^{n-1} \hat{B}]$$


### Quantum Controllability in Closed Systems
I was thinking about the same problem for an ideal quantum system. Here my control parameters linearly change the Hamiltonian:
$$\frac{d \ket{\psi}}{dt} = - i (H_0 + \sum_{i=1}^q H_i u_i(t)) \ket{\psi}$$
This system still has linear equations of motion, but unlike the traditional control theory example, the control is multiplicative. Meaning the control parameters multiply the system parameters. This means the traditional theory above does not apply. It also means that the control parameters $u_i(t)$ clearly cannot arbitrarily set $\ket{\psi}$, for example they cannot change its normalization. Therefore, the definition of controllability it seems should change when using multiplicative control. The most general operation that a time varying Hamiltonian can do to a wavefunction over time T is perform an arbitrary unitary evolution (perform an arbitrary gate). So, it seems to me that an appropriate definition for controllability in this context is for any unitary matrix $\hat{U}$ is there a choice of $\{u_i(t)\}$ such that the evolution operator from $0$ to $T$ is $U$. In fact, you can solve this problem exactly in a similar manor as above. If I take the log of a unitary operator, I always find an anti-Hermitian operator. Importantly, the anti-Hermitian matrices are a linear subspace when the space (treated as a vector space with real coefficients). The evolution of the wavefunction with a time varying Hamiltonian be expressed by a time order exponential:
$$U(T, 0) = \mathcal{T}exp(-i \int_{0}^T H(t) dt) = 1 - i \int_{0}^T H(t_1) dt_1 - \int_{0}^T H(t_2) \int_{0}^{t_2} H(t_1) dt_1 dt_2 + ...$$
There is a series expression for the logarithm of evolution operator known as the Magnus expansion commonly used in Floquet enginery.
$$ln U(T, 0) = -i \int_{0}^{T} H(t_1) dt_1 - \frac{1}{2} \int_{0}^{T} \int_{0}^{t_1} [H(t_1), H(t_2)] dt_2 dt_1 + \frac{i}{6} \int_{0}^{T} \int_{0}^{t_1} \int_{0}^{t_2} ([H(t_1) [H(t_2), H(t_3)]] + [H(t_3) [H(t_2), H(t_1)]]) dt_3 dt_2 dt_1 + ...$$
Assuming that $N$ is the dimension of the Hilbert space, any expressions involving powers of N or greater of $H_0$ or $H_q$ can be reduced to power between $0$ and $N-1$ using their characteristic polynomials. The one thing I still have not been able to figure out is how to prove that strings like $H_0 H_1 H_0 H_1 ...$ do not go on forever. It obvious that they cannot since the vector space it finite dimensional, but I am not sure what the smallest set terms required to specify the full expansion is. The magnus expansion can therefore be written as:
$$ln U(T, 0) = \sum_{i=0}^q c_i(T) H_i + \sum_{i,j=0}^q c_{i,j}(T) [H_i, H_j] + \sum_{i,j,k=0}^q c_{i,j,k}(T) [H_i, [H_j, H_k]] + ...$$
$$ln U(T, 0) = \sum_{i_0, i_1, ..., i_q=0}^{N-1} \tilde{c}_{i_0, i_1, ..., i_q}(T) \Pi_{j=0}^q [(H_j)^{i_j}] + ...$$
This sum has more than $N^q$ terms. In other words, for a set of control parameters to create an arbitrary quantum gate you only need to check whether all the commutators of the control Hamiltonians span a vector space with real coefficients of dimension $N + 2N(N-1)/2 = N^2$ (the dimension of the space of anti-Hermitian matrices as a real vector space). Checking this condition is just a mater of checking the rank of a large matrix. Of course, for any quantum system of decent size this will become completely impractical. The more interesting implication is that it takes a relatively small number control parameters to create an arbitrary quantum gate, so long as their control Hamiltonians do not have trivial commutators. Just one control parameter already leads to $N^2$ terms.

Was this result already known? Is there something related I could read?

Thank you,

Henry Hunt
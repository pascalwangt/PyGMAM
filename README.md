# PyGMAM

The Python 3 script gmam_demo.py found in this repository implements the geometric minimum action method (gMAM) as presented in Heymann, M. and Vanden-Eijnden, E. (2008), "The geometric minimum action method: A least action principleon the space of curves." Communications on Pure and Applied Mathematics

This algorithm uses a steepest descent method to compute the instanton going from an initial state to a target state for SDEs of the form:
dX(t) = b(X(t)) + sigma * dW(t)
where X is the n-dimensional state vector, b is the drift vetor (referred as force_matrix), sigma is the noise matrix (referred noise_matrix) and a = sigma * sigma.T is the diffusion matrix

In gmam_demo.py, the geometric minimum action method is applied to a 2D double well where the instanton is a straight line. A toggle-able plotting option can be used to visualise the successive steps. 

If you want to apply it to any n-dimensional dynamical system, you only need to change the system parameters at the top of the file, under "SDE parameters". You may want to adapt the corresponding plotting lines.

If there are convergence issues, there are usually solved by decreasing the artificial time step delta_tau or providing a better initial guess. Because of the implicit scheme, the artificial time step delta_tau required to achieve stability is independent of the number of trajectory points N.

# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 09:19:44 2019

@author: pasca
"""

import numpy as np
import sympy as sp
import scipy as sci
import matplotlib.pyplot as plt
import matplotlib as mpl

"""
gMAM implementation as presented in:
Heymann, M. and Vanden-Eijnden, E. (2008).
The geometric minimum action method: A least action principleon the space of curves.
Communications on Pure and Applied Mathematics

Solves the instanton going from initial_state to target_state for SDEs of the form:
dX(t) = b(X(t)) + sigma * dW(t)
where b is the drift (referred as force_matrix)
      sigma is the noise matrix (referred noise_matrix) and a = sigma*sigma.T is the diffusion matrix
"""

#SDE parameters
from sympy.abc import x,y               # use sympy symbols
force_matrix = sp.Matrix([x-x**3,-y])   # drift vector (use sympy symbols)
sigma_matrix = sp.eye(2)                # noise matrix (use sympy symbols)
initial_state = np.array([-1, 0])       # starting point of the instanton (use numerical value)
target_state = np.array([1,0])          # end point of the instanton (use numerical value)

#gMAM parameters
N = 1000                                # number of points in the trajectory
delta_tau = 0.1                         # implicit artificial timestep (the algorithm is stable when delta_tau is smaller than a threshold independant of N)
kmax = 70                              # maximum number of iterations
threshold = 0.1                         # stopping criterion: the algorithm stops when norm(phi_i_update-phi_i) < threshold

s = np.linspace(-1, 1, N)
phi_0 = np.array([s, -(s+1)*(s-1)])      #initial guess of trajectory (shape (dimension, N), can be crucial for the convergence) 

#verbose parameters
plot = 1                                #plot output
freq = 1                                #frequency of verbose (printed every freq iterations)
cmap = plt.cm.plasma

#%%
#diffusion matrix
a = sigma_matrix*sigma_matrix.T
a_inv = a**-1
#drift vector
b = force_matrix

#redefine variables
dim = len(force_matrix)

alphabet_x = list(sp.ordered(b.free_symbols))
new_alphabet_x = sp.symbols(f'x:{dim}')
b = b.subs(list(zip(alphabet_x, new_alphabet_x)))
alphabet_x = new_alphabet_x
x = sp.Matrix(new_alphabet_x)           #variable in space

alphabet_p = sp.symbols(f'p:{dim}')
p = sp.Matrix(alphabet_p)               #conjuguate variable of x in Hamiltonian formalism (referred as theta in Heymann, M. and Vanden-Eijnden, E. (2008))


#%% define hamiltonian, its derivatives and other quantities

ham = (b.T*p+1/2*p.T*a*p) #Hamiltonian of the system (scalar)

ham_p = ham.jacobian(p) #gradient with respect to p of the Hamiltonian (vector)
ham_x = ham.jacobian(x) #gradient with respect to x of the Hamiltonian (vector)

ham_px = ham_p.jacobian(x) #Jacobian matrix with respect to x and p (dim x dim matrix) 
ham_pp = ham_p.jacobian(p) #Jacobian matrix with respect to p and p (dim x dim matrix) 

theta = a_inv*(b.T.dot(a_inv*b)/p.T.dot(a_inv*p)*p-b) #referred as theta in  Heymann, M. and Vanden-Eijnden, E. (2008)
lamda = b.T.dot(a_inv*b)/p.T.dot(a_inv*p)             #referred as lambda in  Heymann, M. and Vanden-Eijnden, E. (2008)

#%% convert the sympy sumbolic expressions to numpy functions

ham_p_np = sp.lambdify(alphabet_x+alphabet_p, ham_p, "numpy")
ham_x_np = sp.lambdify(alphabet_x+alphabet_p, ham_x, "numpy")

ham_px_np = sp.lambdify(alphabet_x+alphabet_p, ham_px, "numpy")
ham_pp_np = sp.lambdify(alphabet_x+alphabet_p, ham_pp, "numpy")

ham_pp_x_np = sp.lambdify(alphabet_x+alphabet_p, ham_pp*ham_x.T, "numpy")

theta_np = sp.lambdify(alphabet_x+alphabet_p, theta, "numpy")
lamda_np = sp.lambdify(alphabet_x+alphabet_p, lamda, "numpy")

#%% outer loop

incr = np.inf
k = 1
while k<kmax and incr > threshold:
    
    if k==1:
        phi_i = phi_0
    
    #calculating the numerical derivatives
    phi_i_prime = (phi_i[:,2:]-phi_i[:,:-2])/(2/N) # 1 to N-1
    theta_i = np.squeeze(theta_np(*phi_i[:,1:-1], *phi_i_prime)) #sympy matrix is inherently 2D 
    
    
    lambda_i = np.einsum('ij,ij -> j',  np.squeeze(ham_p_np(*phi_i[:,1:-1], *theta_i)), phi_i_prime) /np.linalg.norm(phi_i_prime, axis = 0)
    lambda_0 = 3*lambda_i[0]-3*lambda_i[1]+lambda_i[2]
    lambda_N = 3*lambda_i[-1]-3*lambda_i[-2]+lambda_i[-3]
    lambda_i_full = np.insert(np.append(lambda_i, lambda_N), 0, lambda_0)
    lambda_i_prime = (lambda_i_full[2:]-lambda_i_full[:-2])/(2/N)
    
    
    # calculating the terms of step (2) in Heymann, M. and Vanden-Eijnden, E. (2008) and putting them into the (dim x N) matrix B
    term_2_i = lambda_i*np.einsum('ijk, ki -> ji', np.array([ham_px_np(*x,*p) for x,p in zip(phi_i[:,1:-1].T, theta_i.T)]), phi_i_prime)
    term_3_i = np.squeeze(ham_pp_x_np(*phi_i[:,1:-1], *theta_i))
    term_4_i = lambda_i*lambda_i_prime*phi_i_prime
    left_hand = phi_i[:,1:-1]/delta_tau
    
    B = np.empty((dim, N))
    B[:,0], B[:,-1], B[:,1:-1] = initial_state, target_state, -term_2_i+term_3_i+term_4_i+left_hand
    
    
    #solving the matrix equation: A phi_tilde =B by taking advantage that A is banded
    diag, band, ab = np.ones(N), np.zeros(N), np.zeros((3, N))
    diag[1:-1]= 1/delta_tau + 2*N**2*lambda_i**2
    band = -lambda_i**2*N**2
    ab[0, 2:], ab[1], ab[2,:-2] =  band, diag, band
    phi_tilde = sci.linalg.solve_banded((1,1), ab, B.T) # (N, dim)

        
    #interpolation reparametrisation step with cubic splines
    tck, u = sci.interpolate.splprep(phi_tilde.T, k=1,s=0)
    func = sci.interpolate.CubicSpline(u, phi_tilde)
    phi_i_update = func(np.linspace(0,1,N)).T  #the points in phi_i_update should be equidistant
    # phi_i_update = np.array(sci.interpolate.splev(np.linspace(0,1,N), tck))   #alternative with B-splines
    
    
    #warning is algorithm is unstable
    if np.linalg.norm(initial_state-phi_i_update[:,0]) > 0.1 or  np.linalg.norm(target_state-phi_i_update[:,-1])>0.1:
        print('WARNING: boundary conditions moves, the algorithm is probably unstable')
        print(f'xA: {phi_i_update[0,0]:.2e}, {phi_i_update[1,0]:.2e} \nxB: {phi_i_update[0,-1]:.2e}, {phi_i_update[1,-1]:.2e} ')
        print('Advice: reduce delta_tau or put a better initial guess')
    
    
    #L^2 difference between previous iterations for the stopping criterion
    incr = np.sum(np.linalg.norm(phi_i_update-phi_i, axis = 0)) 
    phi_i = phi_i_update
    
    #verbose
    if k%freq==0:
        print(f"Iteration {k}: {incr:.1f}") #print L2 norm difference 
    
    #plotting intermediate trajectories
    if plot:
        if k==1:
            plt.figure(3)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.plot(phi_0[0], phi_0[1], color = cmap(k/kmax))
            pass
        
        if k%freq==0:
            fig = plt.figure(3)
            plt.plot(phi_i[0], phi_i[1], color = cmap(k/kmax), linewidth = 0.5)

    k+=1

#the final instanton is stored in phi_i
instanton = phi_i


#plot instanton
if plot:
    fig = plt.figure(3)
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    clb =  mpl.colorbar.ColorbarBase(norm=mpl.colors.Normalize(vmin=0.,vmax=kmax), cmap= cmap, ax=cax)
    clb.ax.set_title('$k$')
    ax.plot(instanton[0], instanton[1], label = f'instanton', linewidth = 2, color = 'red', zorder = 2)
    ax.legend()

    plt.figure(4)
    plt.plot(instanton[0], instanton[1], label = f'instanton', linewidth = 2)
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.ylim(-1,1)

# JIMWLK

JIWMLK evolution in Julia with the goal of writing rc-parent-dipole JIMWLK on CUDA. 


## Initial conditions on the field of Wilson lines

* Consider a Gaussian field $\rho$ with the variance (in lattice unites already) 

$$ \langle \rho^a( x = a i_x) \rho^b( y =a i_y) \rangle = \frac{1}{N_y a^2} \mu^2 \delta^{ab} \delta_{i_x, i_y} $$
where $i_x$, $i_y$, $x$ and $y$ are two-dimensional vectors  
* One needs to solve the Posson equation  $\Delta A^a(x) = \rho^a(x)$
In Fourier space, this is simply: 
$$ A^a(k) = \frac{\rho(k)}{m^2+\frac{4}{a^2} ( \sin^2 (\frac{k_1 a }{ 2 } )  + \sin^2 (\frac{k_2 a }{ 2 } ) )}$$
For periodic boundary conditions $k_{1,2} = \frac{2\pi}{L} i_{1,2}$, where $i_{1,2}$ are the integers $i_{1,2}=0,...,N$ and $L=a \, N$; we also accounted for IR regularization with $m^2$. 
Thus the procedure involves 1) FFT of $\rho$ to momentum sapce; 2) finding $A(k)$; 3) FFT $A$  to coordinate space 
* Next we need to compute path ordered Wilson line; we start with just a single local Wilson line first. 
$$ U(x) = \exp (i t^a A^a(x)), $$ where $U$ is 3 by 3 matrix; $t^a$ are Gell-Mann matrices.    
We use the package https://juliahub.com/ui/Packages/GellMannMatrices/GVbjE/0.1.1 . 
* We repeat the procedure $N_y$ times and (matrix) multiply all $U$ together $U = U_1 U_2 U_3 ...$.   
* The resulting $U(x)$ serves as the initial condition for JIMWLK evolution


## Evolution for fixed coupling

$$  V_{Y+\Delta Y}(x) = \exp \Bigg( - i   \frac{\sqrt{\alpha \Delta Y}}{\pi} \int_z  K_i (x-z) \cdot (V(z) {\xi_i}(z) V^\dagger(z))  \Bigg) V(x) \exp \Bigg(  i   \frac{\sqrt{\alpha \Delta Y}}{\pi} \int_z K_i (x-z) \cdot  \xi_i(z)\Bigg), $$
where $\xi_i(x) = t^a \xi_i^a(x)$  and  $\xi_i^a(x)$ is a random Gaussian noise with unit variance: 

$$ \langle \xi^a_i (x)  \xi^b_j (x') \rangle = \frac{1}{a^2} \delta_{ij}\delta_{ab}  \delta_{i_x, i_y} \,. $$

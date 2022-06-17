# JIMWLK

JIWMLK evolution in Julia with the goal of writing rc-parent-dipole JIMWLK on CUDA. 


## Initial conditions on the field of Wilson lines

* Consider a Gaussian field $\rho$ with the variance (in lattice unites already) 

$$ \langle \rho^a( x = a i_x) \rho^b( y =a i_y) \rangle = \frac{1}{N_y a^2} \mu^2 \delta^{ab} \delta_{i_x, i_y} $$
where $i_x$, $i_y$, $x$ and $y$ are two-dimensional vectors  
* One needs to solve the Posson equation  $\Delta A^a(x) = \rho^a(x)$
*



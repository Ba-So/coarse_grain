# Coarse Graining Routine

This Routine is written for the model output of the ICON-IAP on an unstructured grid.

It's neccessary input are the horizontal velocities u and v, the density and temperature at gridpoints. Furthermore the size of individual gridcells and their longditude and latitude coordinates.

## Intended function:

This routine is supposed to be used to learn about the statistics of local entropy production rates. For now "only" the local entropy procduction rates due to turbulent diffusion are computed. Future work may add the local entropy production rates due to heat diffusion.

### entropy production through turbulent diffusion 

The turbulent diffusion as per Smagorinsky is formulated as follows:

'''
- <img src="http://latex.codecogs.com/gif.latex?%5Cepsilon%20%3D%20%5Chat%7BT%7D%5Csigma"/> 
- <img src="http://latex.codecogs.com/gif.latex?%3D-%28%5Cbar%7B%5Crho%20%5Cvec%7Bv%7D%5E%7B%27%27%7D%5Cvec%7Bv%7D%5E%7B%27%27%7D%7D%29%5Ccdot%20%5Cnabla%20%5Chat%7Bv%7D" /> 
'''

Where the ^ are mass weighted volumetric averages. In modeling the averaged quantities are actually resolved, where the '' fluctuations are unresolved and parametrized. 

### Idea and Function of this project:

In order to learn about the statistics of entropy production rates at a certain resolution, a model run on higher resolution is done and subsequently coarse-grained:

- <img src="https://latex.codecogs.com/gif.latex?\hat{I}=\hat\hat{I} + \hat{I}^{''}"/> 

Where the second ^ is the coarse-grained average. The local entropy production through turbulent diffusion.

- <img src="https://latex.codecogs.com/gif.latex?\hat{epsilon}=\hat\hat{T}\sigma \\ 
-         = -(\bar{\rho \hat{\vec{v}}^{''}\hat{\vec{v}}^{''}})\cdot \nabla\hat\hat(v)" /> 


Whenever coarse_grain is run on a new data set it perpares the gridfile. 


# This Script will take the Information on wind velocities, Temperature and

# 
# density to compute the Dissipation:
# \epsilon = \hat{T} \sigma 
#          = -(\bar{\rho \vec{v}^{''}\vec{v}^{''}})\cdot \nabla\hat(v)
#
# The initial try will only look at horizontal Diffusion, later the full 3D
# Diffusion may be included
#
# \hat(\v) 
#          is the weight averaged velocity over a flexible domain in horizontal
#          orientation
#
# \nabla \hat(v)
#          is a tensor, use center of neighbouring domains to compute correct
#          d/dx and d/dy!! ( I.e Matrix)!
#
# \vec{v}^{''} = \v'
#          will be computed on each grid point contained in the flexible domain
#
# \v'\v' 
#          is a Tensor
#
# \bar{\rho \v'\v'}
#          is the average over the felxible domain, where \rho is the density
#          at each of the respective gird points
#
# \bar{\rho \v'\v'} \cdot \nabla\hat{v} 
#          therefore requires the tensor multiplication
#
# \hat{T} 
#          is the Temperature weight averaged over the flexible domain
#
# \sigma
#          will then be the entropy production over that respective domain
#          some thinking about the physical thing that might be is required.
#          But this might however result in negative entropy production values.
#
# u_tf v_tf
#          are the quantities taken from the ICON output

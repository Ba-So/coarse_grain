# This Script will take the Information on wind velocities, Temperature and
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

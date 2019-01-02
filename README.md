# Coarse Graining Routine

This Routine is written for the model output of the ICON-IAP on an unstructured grid.

It's neccessary input are the horizontal velocities u and v, the density and temperature at gridpoints. Furthermore the size of individual gridcells and their longditude and latitude coordinates.

## Intended function:

This routine is supposed to be used to learn about the statistics of local entropy production rates. For now "only" the local entropy procduction rates due to turbulent diffusion are computed. Future work may add the local entropy production rates due to heat diffusion.

### entropy production through turbulent diffusion 

The turbulent diffusion as per Smagorinsky is formulated as follows:

- <img src="http://latex.codecogs.com/gif.latex?%5Cepsilon%20%3D%20%5Chat%7BT%7D%5Csigma"/> 
- <img src="http://latex.codecogs.com/gif.latex?%5C%5C%20%26%3D-%28%5Coverline%7B%5Crho%20%5Cvec%7Bv%7D%5E%7B%27%27%7D%5Cvec%7Bv%7D%5E%7B%27%27%7D%7D%29%5Ccdot%20%5Cnabla%20%5Chat%7Bv%7D" /> 

Where the ^ are mass weighted volumetric averages. In modeling the averaged quantities are actually resolved, where the '' fluctuations are unresolved and parametrized. 

### Idea and Function of this project:

In order to learn about the statistics of entropy production rates at a certain resolution, a model run on higher resolution is done and subsequently coarse-grained:

- <img src="http://latex.codecogs.com/gif.latex?%5Chat%7BI%7D%20%3D%20%5Chat%7B%5Chat%7BI%7D%7D%20&plus;%20%5Chat%7BI%7D%5E%7B%27%27%7D"/> 

Where the second ^ is the coarse-grained average. The local entropy production through turbulent diffusion.


- <img src="http://latex.codecogs.com/gif.latex?%5Chat%7B%5Cepsilon%7D%20%3D%20%5Chat%7B%5Chat%7BT%7D%7D%5Chat%7B%5Csigma%7D"/> 
- <img src="http://latex.codecogs.com/gif.latex?%3D-%28%5Coverline%7B%5Crho%20%5Chat%7B%5Cvec%7Bv%7D%5E%7B%27%27%7D%7D%5Chat%7B%5Cvec%7Bv%7D%5E%7B%27%27%7D%7D%7D%29%20%5Ccdot%20%5Cnabla%5Chat%7B%5Chat%7Bv%7D%7D" /> 

## run Shematic: 

 to be continued



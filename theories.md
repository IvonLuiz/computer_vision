### :

### Edge detection
Sobel operator: approximation to derivative of image (gray scale). Kernel operations. Normally run Gaussian blur before.
Kernel horizontal = 
[-1  0  1]
[-2  0  2]
[-1  0  1]
Kernel vertical =   
[-1 -2 -1]
[ 0  0  0]
[-1 -2 -1]

Sobel operator
G = $\sqrt{G_x^2 G_y^2}$
orientation = $atan(G_y/G_x)$

#### Canny edge detection
Takes as inputs the outputs from edge detection. Thinning all the edges so they are 1 pixel wide (we want to know where are the edges, not how trick).

Find if the pixel is bigger than it's edge neighbors in magnitudes. This will produce thin edges.
DCGANs-for-3D-object-generation:

DCGANs to create objects in voxel space

network architecture:

discriminator: 3D CNN -> batch norm -> 3D CNN -> batch norm -> 3D CNN -> relu -> output layer

generator: transposed 3D CNN -> Lrelu -> transposed 3D CNN -> Lrelu -> transposed 3D CNN -> relu -> output layer

data:

The data was loaded via a library called binvox r-w. The data was processed from mesh format into .binvox using binvox application. credit for the binvox application: http://www.patrickmin.com/binvox/binvox.bib

Installation:

if you want to run this script yourself then download the ZIP file and then install tensorflow, numpy, and matplotlib then open one of the .off files with the binvox application after downloading from http://www.patrickmin.com/binvox then put the chairs dir in another dir named chairs(this is due to a mistake I made in a previous git commit dont do this with any other dirs just chairs) after this is done run data_process.py and then you can run train.py! btw: this repository is not commercial grade or anything and is not optimized with the latest methods in ML















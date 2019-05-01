# DCGANs-for-3D-object-generation

DCGANs to create objects in voxel space

# network architecture

discriminator: 3D CNN -> batch norm -> 3D CNN -> batch norm -> 3D CNN -> relu -> output layer

generator: transposed 3D CNN -> Lrelu -> transposed 3D CNN -> Lrelu -> transposed 3D CNN -> relu -> output layer

# data

The data was loaded via a library called binvox r-w.
We used the shapenet dataset to get our data.
The data was processed from mesh format into .binvox using binvox application.
credit for the binvox application: http://www.patrickmin.com/binvox/binvox.bib

































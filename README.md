# DCGANs-for-3D-object-generation

DCGANs to create objects in voxel space

# network architecture

discriminator: 3D CNN -> batch norm -> 3D CNN -> batch norm -> 3D CNN -> relu -> output layer

generator: transposed 3D CNN -> Lrelu -> transposed 3D CNN -> Lrelu -> transposed 3D CNN -> relu -> output layer

# Installation
if you want to run this script yourself then download the ZIP file and then install
needed libraries.
After this replace the paths at the top to thier respecive paths in your system
after that run train.py !




























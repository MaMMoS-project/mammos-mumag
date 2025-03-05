"""Standard problem on cubic geometry."""

import sys
import salome
from salome.geom import geomBuilder
import salome_notebook
import SMESH
from salome.smesh import smeshBuilder

salome.salome_init()
notebook = salome_notebook.NoteBook()

### Material parameters
args = sys.argv[1:]
if len(args) == 0:
    print("usage:")
    print("salome -t cube.py args:boxsize,meshsize,layers")
    print("where boxsize  is the size of the cube in nm")
    print("      meshsize is the size of the finite element mesh in nm")
    print("      layers is the number of mesh layers in the shell")
    sys.exit()
else:
    box_size = float(args[0])
    cube_mesh_size = float(args[1])
    layers = int(args[2])
    print("size of cube    (nm) :", box_size)


# parameters for spherical shell transformation:
# Imhoff et al., IEEE TRANSACTIONS ON MAGNETICS, VOL. 26, NO. 5, SEPTEMBER 1990, 1659

# On the number of layers (see page 1661, bottom left):
# "for most applications, it is not necessary
#  to have more than two or three layers of finite elements in the
#  image domain to get accurate results ; a single layer is adequate but
#  only with second order elements. In the part of space corresponding
#  to Omega_in the same mesh has been used for computation" 

R = (box_size / 2.0) * 1.8  # radius of sphere enclosing cube
Rinf = R * 1.8  # out radius of spherical shell

shell_mesh_size = (Rinf - R) / layers

print("shell thickness (nm) :", Rinf - R)
print("cube mesh size  (nm) :", cube_mesh_size)
print("shell mesh size (nm) :", shell_mesh_size)



### GEOM component
geompy = geomBuilder.New()
OO = geompy.MakeVertex(0, 0, 0)
OX = geompy.MakeVectorDXDYDZ(1, 0, 0)
OY = geompy.MakeVectorDXDYDZ(0, 1, 0)
OZ = geompy.MakeVectorDXDYDZ(0, 0, 1)
cube = geompy.MakeBoxDXDYDZ(box_size, box_size, box_size)
geompy.TranslateDXDYDZ(cube, -box_size / 2.0, -box_size / 2.0, -box_size / 2.0)
Sphere_1 = geompy.MakeSphereR(R)
Sphere_2 = geompy.MakeSphereR(Rinf)
Partition_1 = geompy.MakePartition(
    [cube, Sphere_1, Sphere_2], [], [], [], geompy.ShapeType["SOLID"], 0, [], 0
)
a1 = geompy.CreateGroup(Partition_1, geompy.ShapeType["SOLID"])
geompy.UnionIDs(a1, [2])
a2 = geompy.CreateGroup(Partition_1, geompy.ShapeType["SOLID"])
geompy.UnionIDs(a2, [36])
a3 = geompy.CreateGroup(Partition_1, geompy.ShapeType["SOLID"])
geompy.UnionIDs(a3, [46])
[a1, a2, a3] = geompy.GetExistingSubObjects(Partition_1, False)
Auto_group_for_Sub_mesh_1 = geompy.CreateGroup(Partition_1, geompy.ShapeType["SOLID"])
geompy.UnionList(Auto_group_for_Sub_mesh_1, [a1, a2])

geompy.addToStudy(OO, "OO")
geompy.addToStudy(OX, "OX")
geompy.addToStudy(OY, "OY")
geompy.addToStudy(OZ, "OZ")
geompy.addToStudy(cube, "cube")
geompy.addToStudy(Sphere_1, "Sphere_1")
geompy.addToStudy(Sphere_2, "Sphere_2")
geompy.addToStudy(Partition_1, "Partition_1")
geompy.addToStudyInFather(Partition_1, a1, "1")
geompy.addToStudyInFather(Partition_1, a2, "2")
geompy.addToStudyInFather(Partition_1, a3, "3")
geompy.addToStudyInFather( Partition_1, Auto_group_for_Sub_mesh_1, 'Auto_group_for_Sub-mesh_1' )

### SMESH component
smesh = smeshBuilder.New()
Mesh_1 = smesh.Mesh(Partition_1,'Mesh_1')
NETGEN_1D_2D_3D = Mesh_1.Tetrahedron(algo=smeshBuilder.NETGEN_1D2D3D)
NETGEN_3D_Parameters_1 = NETGEN_1D_2D_3D.Parameters()
NETGEN_3D_Parameters_1.SetMinSize( cube_mesh_size )
NETGEN_3D_Parameters_1.SetSecondOrder( 0 )
NETGEN_3D_Parameters_1.SetOptimize( 1 )
NETGEN_3D_Parameters_1.SetFineness( 4 )
NETGEN_3D_Parameters_1.SetChordalError( -1 )
NETGEN_3D_Parameters_1.SetChordalErrorEnabled( 0 )
NETGEN_3D_Parameters_1.SetUseSurfaceCurvature( 1 )
NETGEN_3D_Parameters_1.SetFuseEdges( 1 )
NETGEN_3D_Parameters_1.SetQuadAllowed( 0 )
a1_1 = Mesh_1.GroupOnGeom(a1,'1',SMESH.VOLUME)
a2_1 = Mesh_1.GroupOnGeom(a2,'2',SMESH.VOLUME)
a3_1 = Mesh_1.GroupOnGeom(a3,'3',SMESH.VOLUME)
NETGEN_1D_2D_3D_1 = Mesh_1.Tetrahedron(algo=smeshBuilder.NETGEN_1D2D3D,geom=Auto_group_for_Sub_mesh_1)
NETGEN_3D_Parameters_2 = NETGEN_1D_2D_3D_1.Parameters()
NETGEN_3D_Parameters_2.SetMaxSize( cube_mesh_size )
NETGEN_3D_Parameters_2.SetMinSize( cube_mesh_size )
NETGEN_3D_Parameters_2.SetSecondOrder( 0 )
NETGEN_3D_Parameters_2.SetOptimize( 1 )
NETGEN_3D_Parameters_2.SetFineness( 4 )
NETGEN_3D_Parameters_2.SetChordalError( -1 )
NETGEN_3D_Parameters_2.SetChordalErrorEnabled( 0 )
NETGEN_3D_Parameters_2.SetUseSurfaceCurvature( 1 )
NETGEN_3D_Parameters_2.SetFuseEdges( 1 )
NETGEN_3D_Parameters_2.SetQuadAllowed( 0 )
NETGEN_3D_Parameters_2.SetCheckChartBoundary( 48 )
NETGEN_3D_Parameters_1.SetMaxSize( shell_mesh_size )
NETGEN_3D_Parameters_1.SetCheckChartBoundary( 48 )
isDone = Mesh_1.Compute()
# Mesh_1.CheckCompute()
[ a1_1, a2_1, a3_1 ] = Mesh_1.GetGroups()
Sub_mesh_1 = NETGEN_1D_2D_3D_1.GetSubMesh()

# Exporting
try:
    Mesh_1.ExportUNV(r"cube.unv", 0)
except:
    print("ExportUNV() failed. Invalid file name?")

# Set names of Mesh objects
smesh.SetName(Sub_mesh_1, 'Sub-mesh_1')
smesh.SetName(a2_1, '2')
smesh.SetName(a1_1, '1')
smesh.SetName(NETGEN_3D_Parameters_1, 'NETGEN 3D Parameters_1')
smesh.SetName(Mesh_1.GetMesh(), 'Mesh_1')
smesh.SetName(NETGEN_3D_Parameters_2, 'NETGEN 3D Parameters_2')
smesh.SetName(a3_1, '3')
smesh.SetName(NETGEN_1D_2D_3D.GetAlgorithm(), 'NETGEN 1D-2D-3D')

if salome.sg.hasDesktop():
    salome.sg.updateObjBrowser()

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
    print("salome -t -w1 cube.py args:boxsize,meshsize")
    print("where boxsize  is the size of the cube in nm")
    print("      meshsize is the size of the finite element mesh in nm")
    sys.exit()
else:
    box_size = float(args[0])
    mesh_size = float(args[1])
    print("size of cube (nm) :", box_size)
    print("mesh size    (nm) :", mesh_size)

# parameters for spherical shell transformation:
# Imhoff et al., IEEE TRANSACTIONS ON MAGNETICS, VOL. 26, NO. 5, SEPTEMBER 1990, 1659
R = (box_size / 2.0) * 1.8  # radius of sphere enclosing cube
Rinf = R * 1.8  # out radius of spherical shell

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
[
    Edge_1,
    Edge_2,
    Edge_3,
    Edge_4,
    Edge_5,
    Edge_6,
    Edge_7,
    Edge_8,
    Edge_9,
    Edge_10,
    Edge_11,
    Edge_12,
] = geompy.ExtractShapes(a1, geompy.ShapeType["EDGE"], True)
a2 = geompy.CreateGroup(Partition_1, geompy.ShapeType["SOLID"])
geompy.UnionIDs(a2, [36])
a3 = geompy.CreateGroup(Partition_1, geompy.ShapeType["SOLID"])
geompy.UnionIDs(a3, [46])
[a1, a2, a3] = geompy.GetExistingSubObjects(Partition_1, False)
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
geompy.addToStudyInFather(a1, Edge_1, "Edge_1")
geompy.addToStudyInFather(a1, Edge_2, "Edge_2")
geompy.addToStudyInFather(a1, Edge_3, "Edge_3")
geompy.addToStudyInFather(a1, Edge_4, "Edge_4")
geompy.addToStudyInFather(a1, Edge_5, "Edge_5")
geompy.addToStudyInFather(a1, Edge_6, "Edge_6")
geompy.addToStudyInFather(a1, Edge_7, "Edge_7")
geompy.addToStudyInFather(a1, Edge_8, "Edge_8")
geompy.addToStudyInFather(a1, Edge_9, "Edge_9")
geompy.addToStudyInFather(a1, Edge_10, "Edge_10")
geompy.addToStudyInFather(a1, Edge_11, "Edge_11")
geompy.addToStudyInFather(a1, Edge_12, "Edge_12")

### SMESH component
smesh = smeshBuilder.New()
Mesh_1 = smesh.Mesh(Partition_1, "Mesh_1")
NETGEN_1D_2D_3D = Mesh_1.Tetrahedron(algo=smeshBuilder.NETGEN_1D2D3D)
NETGEN_3D_Parameters_1 = NETGEN_1D_2D_3D.Parameters()
NETGEN_3D_Parameters_1.SetSecondOrder(0)
NETGEN_3D_Parameters_1.SetOptimize(1)
NETGEN_3D_Parameters_1.SetChordalError(-1)
NETGEN_3D_Parameters_1.SetChordalErrorEnabled(0)
NETGEN_3D_Parameters_1.SetUseSurfaceCurvature(1)
NETGEN_3D_Parameters_1.SetFuseEdges(1)
NETGEN_3D_Parameters_1.SetQuadAllowed(0)
NETGEN_3D_Parameters_1.SetLocalSizeOnShape(Edge_10, mesh_size)
NETGEN_3D_Parameters_1.SetLocalSizeOnShape(Edge_11, mesh_size)
NETGEN_3D_Parameters_1.SetLocalSizeOnShape(Edge_2, mesh_size)
NETGEN_3D_Parameters_1.SetLocalSizeOnShape(Edge_3, mesh_size)
NETGEN_3D_Parameters_1.SetLocalSizeOnShape(Edge_5, mesh_size)
NETGEN_3D_Parameters_1.SetLocalSizeOnShape(Edge_6, mesh_size)
NETGEN_3D_Parameters_1.SetLocalSizeOnShape(Edge_7, mesh_size)
NETGEN_3D_Parameters_1.SetLocalSizeOnShape(Edge_8, mesh_size)
NETGEN_3D_Parameters_1.SetMaxSize(mesh_size)
NETGEN_3D_Parameters_1.SetMinSize(mesh_size)
NETGEN_3D_Parameters_1.SetFineness(4)
NETGEN_3D_Parameters_1.SetCheckChartBoundary(32)
a1_1 = Mesh_1.GroupOnGeom(a1, "1", SMESH.VOLUME)
a2_1 = Mesh_1.GroupOnGeom(a2, "2", SMESH.VOLUME)
a3_1 = Mesh_1.GroupOnGeom(a3, "3", SMESH.VOLUME)
isDone = Mesh_1.Compute()
[a1_1, a2_1, a3_1] = Mesh_1.GetGroups()

# Exporting
try:
    Mesh_1.ExportUNV(r"cube.unv", 0)
except:
    print("ExportUNV() failed. Invalid file name?")

# Set names of Mesh objects
smesh.SetName(NETGEN_1D_2D_3D.GetAlgorithm(), "NETGEN 1D-2D-3D")
smesh.SetName(NETGEN_3D_Parameters_1, "NETGEN 3D Parameters_1")
smesh.SetName(Mesh_1.GetMesh(), "Mesh_1")
smesh.SetName(a3_1, "3")
smesh.SetName(a2_1, "2")
smesh.SetName(a1_1, "1")

if salome.sg.hasDesktop():
    salome.sg.updateObjBrowser()

#!/usr/bin/env python

###
### This file is generated automatically by SALOME v9.12.0 with dump python functionality
###

import sys
import salome

salome.salome_init()
import salome_notebook
notebook = salome_notebook.NoteBook()
#sys.path.insert(0, r'C:/Users/tschr/home/W2/mammos/suite/meshing')
sys.path.insert(0, r'/home/tom/home/W2/mammos/suite/meshing')

###
### GEOM component
###

import GEOM
from salome.geom import geomBuilder
import math
import SALOMEDS


geompy = geomBuilder.New()

O = geompy.MakeVertex(0, 0, 0)
OX = geompy.MakeVectorDXDYDZ(1, 0, 0)
OY = geompy.MakeVectorDXDYDZ(0, 1, 0)
OZ = geompy.MakeVectorDXDYDZ(0, 0, 1)
cube = geompy.MakeBoxDXDYDZ(60, 60, 60)
geompy.TranslateDXDYDZ(cube, -30, -30, -30)
Sphere_1 = geompy.MakeSphereR(55)
Sphere_2 = geompy.MakeSphereR(85)
Partition_1 = geompy.MakePartition([cube, Sphere_1, Sphere_2], [], [], [], geompy.ShapeType["SOLID"], 0, [], 0)
a1 = geompy.CreateGroup(Partition_1, geompy.ShapeType["SOLID"])
geompy.UnionIDs(a1, [2])
[Edge_1,Edge_2,Edge_3,Edge_4,Edge_5,Edge_6,Edge_7,Edge_8,Edge_9,Edge_10,Edge_11,Edge_12] = geompy.ExtractShapes(a1, geompy.ShapeType["EDGE"], True)
a2 = geompy.CreateGroup(Partition_1, geompy.ShapeType["SOLID"])
geompy.UnionIDs(a2, [36])
a3 = geompy.CreateGroup(Partition_1, geompy.ShapeType["SOLID"])
geompy.UnionIDs(a3, [46])
[a1, a2, a3] = geompy.GetExistingSubObjects(Partition_1, False)
geompy.addToStudy( O, 'O' )
geompy.addToStudy( OX, 'OX' )
geompy.addToStudy( OY, 'OY' )
geompy.addToStudy( OZ, 'OZ' )
geompy.addToStudy( cube, 'cube' )
geompy.addToStudy( Sphere_1, 'Sphere_1' )
geompy.addToStudy( Sphere_2, 'Sphere_2' )
geompy.addToStudy( Partition_1, 'Partition_1' )
geompy.addToStudyInFather( Partition_1, a1, '1' )
geompy.addToStudyInFather( Partition_1, a2, '2' )
geompy.addToStudyInFather( Partition_1, a3, '3' )
geompy.addToStudyInFather( a1, Edge_1, 'Edge_1' )
geompy.addToStudyInFather( a1, Edge_2, 'Edge_2' )
geompy.addToStudyInFather( a1, Edge_3, 'Edge_3' )
geompy.addToStudyInFather( a1, Edge_4, 'Edge_4' )
geompy.addToStudyInFather( a1, Edge_5, 'Edge_5' )
geompy.addToStudyInFather( a1, Edge_6, 'Edge_6' )
geompy.addToStudyInFather( a1, Edge_7, 'Edge_7' )
geompy.addToStudyInFather( a1, Edge_8, 'Edge_8' )
geompy.addToStudyInFather( a1, Edge_9, 'Edge_9' )
geompy.addToStudyInFather( a1, Edge_10, 'Edge_10' )
geompy.addToStudyInFather( a1, Edge_11, 'Edge_11' )
geompy.addToStudyInFather( a1, Edge_12, 'Edge_12' )

###
### SMESH component
###

import  SMESH, SALOMEDS
from salome.smesh import smeshBuilder

smesh = smeshBuilder.New()
#smesh.SetEnablePublish( False ) # Set to False to avoid publish in study if not needed or in some particular situations:
                                 # multiples meshes built in parallel, complex and numerous mesh edition (performance)

Mesh_1 = smesh.Mesh(Partition_1,'Mesh_1')
NETGEN_1D_2D_3D = Mesh_1.Tetrahedron(algo=smeshBuilder.NETGEN_1D2D3D)
NETGEN_3D_Parameters_1 = NETGEN_1D_2D_3D.Parameters()
NETGEN_3D_Parameters_1.SetSecondOrder( 0 )
NETGEN_3D_Parameters_1.SetOptimize( 1 )
NETGEN_3D_Parameters_1.SetChordalError( -1 )
NETGEN_3D_Parameters_1.SetChordalErrorEnabled( 0 )
NETGEN_3D_Parameters_1.SetUseSurfaceCurvature( 1 )
NETGEN_3D_Parameters_1.SetFuseEdges( 1 )
NETGEN_3D_Parameters_1.SetQuadAllowed( 0 )
NETGEN_3D_Parameters_1.SetLocalSizeOnShape(Edge_10, 1.4)
NETGEN_3D_Parameters_1.SetLocalSizeOnShape(Edge_11, 1.4)
NETGEN_3D_Parameters_1.SetLocalSizeOnShape(Edge_2, 1.4)
NETGEN_3D_Parameters_1.SetLocalSizeOnShape(Edge_3, 1.4)
NETGEN_3D_Parameters_1.SetLocalSizeOnShape(Edge_5, 1.4)
NETGEN_3D_Parameters_1.SetLocalSizeOnShape(Edge_6, 1.4)
NETGEN_3D_Parameters_1.SetLocalSizeOnShape(Edge_7, 1.4)
NETGEN_3D_Parameters_1.SetLocalSizeOnShape(Edge_8, 1.4)
NETGEN_3D_Parameters_1.SetMaxSize( 4.8 )
NETGEN_3D_Parameters_1.SetMinSize( 1.4 )
NETGEN_3D_Parameters_1.SetFineness( 4 )
NETGEN_3D_Parameters_1.SetCheckChartBoundary( 32 )
a1_1 = Mesh_1.GroupOnGeom(a1,'1',SMESH.VOLUME)
a2_1 = Mesh_1.GroupOnGeom(a2,'2',SMESH.VOLUME)
a3_1 = Mesh_1.GroupOnGeom(a3,'3',SMESH.VOLUME)
isDone = Mesh_1.Compute()
[ a1_1, a2_1, a3_1 ] = Mesh_1.GetGroups()
try:
  # Mesh_1.ExportUNV( r'C:/Users/tschr/home/W2/mammos/suite/meshing/cube.unv', 0 )
  Mesh_1.ExportUNV( r'cube.unv', 0 )
  pass
except:
  print('ExportUNV() failed. Invalid file name?')


## Set names of Mesh objects
smesh.SetName(NETGEN_1D_2D_3D.GetAlgorithm(), 'NETGEN 1D-2D-3D')
smesh.SetName(NETGEN_3D_Parameters_1, 'NETGEN 3D Parameters_1')
smesh.SetName(Mesh_1.GetMesh(), 'Mesh_1')
smesh.SetName(a3_1, '3')
smesh.SetName(a2_1, '2')
smesh.SetName(a1_1, '1')


if salome.sg.hasDesktop():
  salome.sg.updateObjBrowser()

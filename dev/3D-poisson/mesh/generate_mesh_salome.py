#!/usr/bin/env python

###
### This file is generated automatically by SALOME v9.14.0 with dump python functionality
###

import pathlib
import sys
import salome

salome.salome_init()
import salome_notebook
notebook = salome_notebook.NoteBook()
sys.path.insert(0, str(pathlib.Path(".").absolute()))

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
Sphere_1 = geompy.MakeSphereR(2)
Sphere_2 = geompy.MakeSphereR(3)
Partition_1 = geompy.MakePartition([Sphere_1, Sphere_2], [], [], [], geompy.ShapeType["SOLID"], 0, [], 0)
core = geompy.CreateGroup(Partition_1, geompy.ShapeType["SOLID"])
geompy.UnionIDs(core, [2])
shell = geompy.CreateGroup(Partition_1, geompy.ShapeType["SOLID"])
geompy.UnionIDs(shell, [11])
boundary = geompy.CreateGroup(Partition_1, geompy.ShapeType["FACE"])
geompy.UnionIDs(boundary, [13])
[core, shell, boundary] = geompy.GetExistingSubObjects(Partition_1, False)
geompy.addToStudy( O, 'O' )
geompy.addToStudy( OX, 'OX' )
geompy.addToStudy( OY, 'OY' )
geompy.addToStudy( OZ, 'OZ' )
geompy.addToStudy( Sphere_1, 'Sphere_1' )
geompy.addToStudy( Sphere_2, 'Sphere_2' )
geompy.addToStudy( Partition_1, 'Partition_1' )
geompy.addToStudyInFather( Partition_1, core, 'core' )
geompy.addToStudyInFather( Partition_1, shell, 'shell' )
geompy.addToStudyInFather( Partition_1, boundary, 'boundary' )

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
NETGEN_3D_Parameters_1.SetMaxSize( 0.2 )
NETGEN_3D_Parameters_1.SetMinSize( 0.047412 )
NETGEN_3D_Parameters_1.SetSecondOrder( 0 )
NETGEN_3D_Parameters_1.SetOptimize( 1 )
NETGEN_3D_Parameters_1.SetFineness( 2 )
NETGEN_3D_Parameters_1.SetChordalError( -1 )
NETGEN_3D_Parameters_1.SetChordalErrorEnabled( 0 )
NETGEN_3D_Parameters_1.SetUseSurfaceCurvature( 1 )
NETGEN_3D_Parameters_1.SetFuseEdges( 1 )
NETGEN_3D_Parameters_1.SetQuadAllowed( 0 )
NETGEN_3D_Parameters_1.SetCheckChartBoundary( 0 )
core_1 = Mesh_1.GroupOnGeom(core,'core',SMESH.VOLUME)
shell_1 = Mesh_1.GroupOnGeom(shell,'shell',SMESH.VOLUME)
boundary_1 = Mesh_1.GroupOnGeom(boundary,'boundary',SMESH.FACE)
isDone = Mesh_1.Compute()
Mesh_1.CheckCompute()
[ core_1, shell_1, boundary_1 ] = Mesh_1.GetGroups()


## Set names of Mesh objects
smesh.SetName(shell_1, 'shell')
smesh.SetName(core_1, 'core')
smesh.SetName(NETGEN_3D_Parameters_1, 'NETGEN 3D Parameters_1')
smesh.SetName(Mesh_1.GetMesh(), 'Mesh_1')
smesh.SetName(boundary_1, 'boundary')
smesh.SetName(NETGEN_1D_2D_3D.GetAlgorithm(), 'NETGEN 1D-2D-3D')


if salome.sg.hasDesktop():
  salome.sg.updateObjBrowser()

Mesh_1.ExportMED("Omega.med")

#!/usr/bin/env python

###
### This file is generated automatically by SALOME v9.14.0 with dump python functionality
###

import sys
import salome

salome.salome_init()
import salome_notebook
notebook = salome_notebook.NoteBook()
sys.path.insert(0, r'/home/petrocch/repo/mammos/mammos-mumag/dev/2D-poisson')

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
Disk_1 = geompy.MakeDiskR(2, 1)
Disk_2 = geompy.MakeDiskR(3, 1)
Partition_1 = geompy.MakePartition([Disk_1, Disk_2], [], [], [], geompy.ShapeType["FACE"], 0, [], 0)
core = geompy.CreateGroup(Partition_1, geompy.ShapeType["FACE"])
geompy.UnionIDs(core, [2])
ring = geompy.CreateGroup(Partition_1, geompy.ShapeType["FACE"])
geompy.UnionIDs(ring, [6])
boundary = geompy.CreateGroup(Partition_1, geompy.ShapeType["EDGE"])
geompy.UnionIDs(boundary, [8])
[core, ring, boundary] = geompy.GetExistingSubObjects(Partition_1, False)
geompy.addToStudy( O, 'O' )
geompy.addToStudy( OX, 'OX' )
geompy.addToStudy( OY, 'OY' )
geompy.addToStudy( OZ, 'OZ' )
geompy.addToStudy( Disk_2, 'Disk_2' )
geompy.addToStudy( Disk_1, 'Disk_1' )
geompy.addToStudy( Partition_1, 'Partition_1' )
geompy.addToStudyInFather( Partition_1, core, 'core' )
geompy.addToStudyInFather( Partition_1, ring, 'ring' )
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
NETGEN_1D_2D = Mesh_1.Triangle(algo=smeshBuilder.NETGEN_1D2D)
core_1 = Mesh_1.GroupOnGeom(core,'core',SMESH.FACE)
ring_1 = Mesh_1.GroupOnGeom(ring,'ring',SMESH.FACE)
boundary_1 = Mesh_1.GroupOnGeom(boundary,'boundary',SMESH.EDGE)
[ core_1, ring_1, boundary_1 ] = Mesh_1.GetGroups()
NETGEN_2D_Parameters_1 = NETGEN_1D_2D.Parameters()
NETGEN_2D_Parameters_1.SetMaxSize( 0.2 )
NETGEN_2D_Parameters_1.SetMinSize( 0.02 )
NETGEN_2D_Parameters_1.SetSecondOrder( 0 )
NETGEN_2D_Parameters_1.SetOptimize( 1 )
NETGEN_2D_Parameters_1.SetFineness( 2 )
NETGEN_2D_Parameters_1.SetChordalError( -1 )
NETGEN_2D_Parameters_1.SetChordalErrorEnabled( 0 )
NETGEN_2D_Parameters_1.SetUseSurfaceCurvature( 1 )
NETGEN_2D_Parameters_1.SetFuseEdges( 1 )
NETGEN_2D_Parameters_1.SetUseDelauney( 0 )
NETGEN_2D_Parameters_1.SetQuadAllowed( 0 )
NETGEN_2D_Parameters_1.SetWorstElemMeasure( 21971 )
NETGEN_2D_Parameters_1.SetCheckChartBoundary( 0 )
isDone = Mesh_1.Compute()
Mesh_1.CheckCompute()
[ core_1, ring_1, boundary_1 ] = Mesh_1.GetGroups()


## Set names of Mesh objects
smesh.SetName(core_1, 'core')
smesh.SetName(Mesh_1.GetMesh(), 'Mesh_1')
smesh.SetName(NETGEN_2D_Parameters_1, 'NETGEN 2D Parameters_1')
smesh.SetName(ring_1, 'ring')
smesh.SetName(NETGEN_1D_2D.GetAlgorithm(), 'NETGEN 1D-2D')
smesh.SetName(boundary_1, 'boundary')


if salome.sg.hasDesktop():
  salome.sg.updateObjBrowser()

Mesh_1.ExportMED("./disk.med")

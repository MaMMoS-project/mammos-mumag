#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G                # MB
#SBATCH --time=1-12             # MINUTES, DAYS-HOURS

#SBATCH --job-name=small
#
# Number of GPUs
#SBATCH --gres=gpu:1
#
#SBATCH --export=NONE
# By default all environment variables of the shell invoking the sbatch command are propagated.
# This may cause unexpected behaviour as for example $HOME used in this script might be different
# from the expected /home/<username>. Also consider $PATH, $OCL* or $CUDA* variables. I recommend
# to set --export=NONE to avoid the propagation.

# show some information
SIMDIR=$HOME/slurm_$SLURM_JOB_ID
echo "JOBNAME:" $SLURM_JOB_NAME
echo "SUBMIT HOST:" $SLURM_SUBMIT_HOST
echo "ALLOC. NODES:" $SLURMD_NODENAME
echo "SUBMIT DIR:" $SLURM_SUBMIT_DIR
echo "SIMDIR:" $SIMDIR

# set variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MAMBA_ROOT_PREFIX='/scandium/home/programs/micromamba/micromamba'
export MAMBA_EXE='/scandium/home/programs/micromamba/bin/micromamba'
PROGRAMS='/scandium/home/programs/UB22'
TOFLY='/scandium/home/programs/UB22/mmag/examples/demagnetization/tofly3'
PY='/scandium/home/programs/UB22/mmag/src/mmag/sim_scripts'
# use this line for meshgems
# SALOME='/scandium/home/programs/UB22/SALOME-9.7.0-native-DB10-SRC/salome'
# use this line for netgen
SALOME='/scandium/home/programs/UB22/SALOME-9.10.0-native-UB22.04-SRC/salome'


# create working directory

MODELNAME="cube"

mkdir -p $SIMDIR
cd $SIMDIR
echo "WORKING DIR:" $PWD

# copy intrinsic properties (.krn) and
# control parameters (.p2)
cp $SLURM_SUBMIT_DIR/$MODELNAME.krn .
cp $SLURM_SUBMIT_DIR/$MODELNAME.p2 .

# copy mesh script
cp $SLURM_SUBMIT_DIR/$MODELNAME.py .

# create a mesh and convert to fly
$SALOME -t -w1 $MODELNAME.py args:40,2,4
$TOFLY -e 1,2 $MODELNAME.unv $MODELNAME.fly

# or use a script that meshes and creates the fly file
# $SALOME -t  salomescript.py

# compute demagnetization curve
$MAMBA_EXE run -n escript5 run-escript $PY/loop.py $MODELNAME

# copy files back if necessary
cp *.vtu $SLURM_SUBMIT_DIR
cp *.dat $SLURM_SUBMIT_DIR
cd ..
rm -r $SIMDIR


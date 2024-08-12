import numpy as np
import os
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({'font.size': 14})


salome  = '/home/tom/pd/SALOME-9.12.0/salome'
tofly   = '/home/tom/home/W2/mammos/suite/hystmag-docker-file/mumag/pymag/py/tofly3'
escript = '/home/tom/pd/escript/bin/run-escript'  

mu0 = 4e-7*np.pi
Js  = 1.05    # T     mu0Ms
A   = 1.3e-11 # J/m   exchange constant
Km  = 0.5*Js*Js/mu0
lex = np.sqrt(A/Km)
Ku  = 0.1*Km

def write_krn(name,K1,Js,A):
  with open(name+'.krn', 'w') as f:
    f.write(f'0.0 0.0 {K1:.8e} 0.0 {Js} {A}\n')
    f.write( '0.0 0.0 0.0  0.0 0.0  0.0\n')    
    f.write( '0.0 0.0 0.0  0.0 0.0  0.0\n')    

def write_p2(name,mz=1.0):
  with open(name+'.p2', 'w') as f:
    s = f'''[mesh]
size = 1.e-9
scale = 0.0

[initial state]
mx = 0.
my = 0.
mz = {mz}

[field]
hstart = 0.0
hfinal = 0.0
hstep = -0.02
hx = 0.
hy = 0.
hz = 1.

[minimizer]
tol_fun = 1e-12
tol_hmag_factor = 1
truncation = 5
precond_iter = 4
verbose = 1\n'''
    f.write(s)
    
def read_last_float(filename):
  with open(filename, 'r') as file:
    lines = file.readlines()
    last_line = lines[-1]
    last_column_value = last_line.split()[-1]
    return float(last_column_value)

def convert_energies(e):
  return (e/mu0 + Ku)/Km
  
Path("results").mkdir(exist_ok=True)  
 
lengths = np.linspace(8.4,8.7,5) 
energies_flower = []
energies_vortex = []
for s in lengths:
    
  cmd = f'{salome} -t -w1 cube.py args:{s*lex*1e9:.2f},{lex*1e9/4:.2f}'
  print(cmd)
  os.system(cmd)
  
  cmd = f'{tofly} -e 1,2 cube.unv cube.fly'
  print(cmd)
  os.system(cmd)

  write_krn('cube',Ku,Js,A)
  
  print("FLOWER")
  write_p2('cube')
  cmd = f'{escript} -t 2 ../../py/loop.py cube'
  print(cmd)
  os.system(cmd)
  energies_flower.append( read_last_float('cube.dat') )
  os.rename('cube.dat',Path('results')/f'cube_flower_{s:.2f}.dat')
  os.rename('cube.0001.vtu',Path('results')/f'cube.0001_flower_{s:.2f}.vtu')

  print("VORTEX")
  write_p2('cube',mz=0.0)
  cmd = f'{escript} -t 2 ../../py/loop.py cube'
  print(cmd)
  os.system(cmd)
  energies_vortex.append( read_last_float('cube.dat') )
  os.rename('cube.dat',Path('results')/f'cube_vortex_{s:.2f}.dat')
  os.rename('cube.0001.vtu',Path('results')/f'cube.0001_vortex_{s:.2f}.vtu')


# plot results
  
energies_flower = np.array(energies_flower)
energies_vortex = np.array(energies_vortex)
energies_flower = convert_energies(energies_flower)
energies_vortex = convert_energies(energies_vortex)

plt.plot(lengths,energies_flower,marker='o',label='flower',c='b')
plt.plot(lengths,energies_vortex,marker='s',label='vortex',c='g')
plt.xlabel(r'size (l$_\mathrm{ex}$)')
plt.ylabel(r'energy density ($\mu_0 M_\mathrm{s}/2$)')
plt.legend()
plt.tight_layout()
plt.savefig(Path('results')/'energies.png')

# clean directory

os.remove('cube.unv')
os.remove('cube.fly')
os.remove('cube.krn')
os.remove('cube.p2')

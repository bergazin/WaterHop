from WaterTranslationRotation import WaterTranslationRotationMove
from blues.engine import MoveEngine
from simulation import *
import json
from blues.settings import *

opt = Settings('blues_cuda.yaml').asDict()
structure = opt['Structure']
print(json.dumps(opt, sort_keys=True, indent=2, skipkeys=True, default=str))

# Adjust the size of the box
structure.box = np.array([ 30.23,  29.52,  95.8846,  90.,      90.,      90.,    ])

# Move 4 waters to the center
structure.atoms[6518].xx = np.array(18.0208001)
structure.atoms[6518].xy = np.array(16.2210011)
structure.atoms[6519].xx = np.array(19.1660004)
structure.atoms[6519].xy = np.array(16.2210011)
structure.atoms[6520].xx = np.array(17.9679995)
structure.atoms[6520].xy = np.array(18.1479988)

structure.atoms[6509].xz = np.array(40.0208001)
structure.atoms[6510].xz = np.array(40.0208001)
structure.atoms[6511].xz = np.array(40.0208001)

structure.atoms[6512].xz = np.array(40.0208001)
structure.atoms[6513].xz = np.array(40.0208001)
structure.atoms[6514].xz = np.array(40.0208001)

structure.atoms[6515].xz = np.array(40.0208001)
structure.atoms[6516].xz = np.array(40.0208001)
structure.atoms[6517].xz = np.array(40.0208001)

# Select last water in the system to act as the protein to be used for distance calculations
import mdtraj as md
wat = md.load('oneWat.pdb')
protein_atoms = wat.topology.select('resid 1916')

#Select move type
ligand = WaterTranslationRotationMove(structure, protein_atoms, water_name='WAT')

#Iniitialize object that selects movestep
ligand_mover = MoveEngine(ligand)

#Generate the openmm.Systems outside SimulationFactory to allow modifications
systems = SystemFactory(structure, ligand.atom_indices, opt['system'])

#Freeze atoms in the alchemical system
systems.md = systems.freeze_atoms(structure, systems.md, **opt['freeze'])
systems.alch = systems.freeze_atoms(structure, systems.alch, **opt['freeze'])

#Generate the OpenMM Simulations
simulations = SimulationFactory(systems, ligand_mover, opt['simulation'], opt['md_reporters'], opt['ncmc_reporters'])

#Energy minimize system
simulations.md.minimizeEnergy(maxIterations=0)
#simulations.md.step(10000)
state = simulations.md.context.getState(getPositions=True, getEnergy=True)
print('Minimized energy = {}'.format(state.getPotentialEnergy().in_units_of(unit.kilocalorie_per_mole)))

#MD simulation
#simulations.md.step(opt['simulation']['nstepsMD'])

# Run BLUES Simulation
blues = BLUESSimulation(simulations)
blues.run(**opt['simulation'])

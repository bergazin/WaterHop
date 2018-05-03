from __future__ import print_function
from WaterHop import WaterHop
from blues.engine import MoveEngine
from blues import utils
from openmmtools import testsystems
from blues.simulation import Simulation, SimulationFactory
import parmed
from simtk import openmm
from simtk import unit
from optparse import OptionParser
import sys
import logging
from blues.reporters import init_logger, BLUESHDF5Reporter, BLUESStateDataReporter
import numpy as np
from simtk.openmm import app

def runNCMC(platform_name, nstepsNC, nprop, outfname):
    #Generate the ParmEd Structure
    prmtop = '/home/bergazin/WaterHop/water/input_files/wall/oneWat.prmtop'
    inpcrd = '/home/bergazin/WaterHop/water/input_files/wall/oneWat.inpcrd'
    structure = parmed.load_file(prmtop, xyz=inpcrd)
    print('Structure: %s' % structure.topology)

    #Define some options
    opt = { 'temperature' : 300.0, 'friction' : 1, 'dt' : 0.001,
            'nIter' : 50000000, 'nstepsNC' : 2500, 'nstepsMD' : 500000, 'nprop' : 5,
            'nonbondedMethod' : 'PME', 'nonbondedCutoff': 10,
            'constraints': 'HBonds', 'freeze_distance' : 0.0,
            'trajectory_interval' : 1000, 'reporter_interval' : 1000,
            'ncmc_traj' : None, 'write_move' : False,
            'platform' : platform_name,
            'outfname' : 'walls_md',
            'verbose' : True}
    logger = init_logger(logging.getLogger(), level=logging.INFO, outfname=opt['outfname'])

    structure.atoms[6518].xx = np.array(18.0208001)
    structure.atoms[6518].xy = np.array(16.2210011)
    structure.atoms[6519].xx = np.array(19.1660004)
    structure.atoms[6519].xy = np.array(16.2210011)
    structure.atoms[6520].xx = np.array(17.9679995)
    structure.atoms[6520].xy = np.array(18.1479988)

    system = structure.createSystem(nonbondedMethod=app.PME,
                            nonbondedCutoff=10*unit.angstroms,
                            constraints=app.HBonds)


    import mdtraj as md
    wat = md.load('/home/bergazin/WaterHop/water/input_files/wall/oneWat.pdb')
    protein_atoms = wat.topology.select('resid 1916')
    #print("protein",protein_atoms)

    # set the masses of the carbon walls and the last water in the system to 0 to hold it in place

    freeze_atoms = wat.topology.select('resname WAL or resid 1916')
    for index in freeze_atoms:
       index = int(index)
       print('Before:', index, system.getParticleMass(index))
       system.setParticleMass(int(index), 0*unit.dalton)
       print('After:', index, system.getParticleMass(index))

    integrator = openmm.LangevinIntegrator(300*unit.kelvin,
                                        1/unit.picosecond,
                                        0.002*unit.picoseconds)
    simulation = app.Simulation(structure.topology, system, integrator)

    #Energy minimize system
    #simulations.md.minimizeEnergy(maxIterations=5000)
    #simulations.md.step(10000)
    #state = simulations.md.context.getState(getPositions=True, getEnergy=True)
    #print('Minimized energy = {}'.format(state.getPotentialEnergy().in_units_of(unit.kilocalorie_per_mole)))

    # Add reporters to MD simulation.
    traj_reporter = openmm.app.DCDReporter(outfname+'-pureMD{}.dcd'.format(nstepsNC), opt['trajectory_interval'])
    progress_reporter = openmm.app.StateDataReporter(sys.stdout, separator="\t",
                               reportInterval=opt['reporter_interval'],
                               step=True, totalSteps=opt['nIter']*opt['nstepsMD'],
                               time=True, speed=True, progress=True, remainingTime=True)
    simulation.reporters.append(traj_reporter)
    simulation.reporters.append(progress_reporter)

    simulation.context.setPositions(structure.positions)
    simulation.context.setVelocitiesToTemperature(500*unit.kelvin)
    simulation.step(opt['nstepsMD'])

parser = OptionParser()
parser.add_option('-f', '--force', action='store_true', default=False,
                  help='run BLUES example without GPU platform')
parser.add_option('-n','--ncmc', dest='nstepsNC', type='int', default=5000,
                  help='number of NCMC steps')
parser.add_option('-p','--nprop', dest='nprop', type='int', default=5,
                  help='number of propgation steps')
parser.add_option('-o','--output', dest='outfname', type='str', default="blues",
                  help='Filename for output DCD')
(options, args) = parser.parse_args()



platformNames = [openmm.Platform.getPlatform(i).getName() for i in range(openmm.Platform.getNumPlatforms())]
if 'CUDA' in platformNames:
    runNCMC('CUDA', options.nstepsNC, options.nprop, options.outfname)
elif 'OpenCL' in platformNames:
    runNCMC('OpenCL',options.nstepsNC, options.nprop, options.outfname)
else:
    if options.force:
        runNCMC('CPU', options.nstepsNC, options.outfname)
    else:
        print('WARNING: Could not find a valid CUDA/OpenCL platform. BLUES is not recommended on CPUs.')
        print("To run on CPU: 'python blues/example.py -f'")

from __future__ import print_function
from WaterTranslationRotation import WaterTranslationRotationMove
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


def runNCMC(platform_name, nstepsNC, nprop, outfname):
    #Generate the ParmEd Structure
    prmtop = '/home/bergazin/WaterHop/water/input_files/onlyWaterBox/BOX1.prmtop'
    inpcrd = '/home/bergazin/WaterHop/water/input_files/onlyWaterBox/BOX1.inpcrd'
    structure = parmed.load_file(prmtop, xyz=inpcrd)
    print('Structure: %s' % structure.topology)

    #Define some options
    opt = { 'temperature' : 300.0, 'friction' : 1, 'dt' : 0.002,
            'nIter' : 100000, 'nstepsNC' : 5000, 'nstepsMD' : 5000, 'nprop' : 5,
            'nonbondedMethod' : 'PME', 'nonbondedCutoff': 10,
            'constraints': 'HBonds',
            'trajectory_interval' : 1000, 'reporter_interval' : 1000,
            'ncmc_traj' : None, 'write_move' : False,
            'platform' : platform_name,
            'outfname' : 'testing',
            'verbose' : True}

    import mdtraj as md
    wat = md.load('/home/bergazin/WaterHop/water/input_files/onlyWaterBox/BOX1.pdb')
    water_atom = wat.topology.select('resid 1')
    print(water_atom)

    #Define the 'model' object we are perturbing here.
    # Calculate particle masses of object to be moved
    water = WaterTranslationRotationMove(structure, water_atom, water_name='WAT')
    water.topology = structure.topology
    water.positions = structure.positions

	# Initialize object that proposes moves.
    mover = MoveEngine(water)

    # Generate the MD, NCMC, ALCHEMICAL Simulation objects
    simulations = SimulationFactory(structure, mover, **opt)
    #system = simulations.generateSystem(structure, **opt)
    simulations.createSimulationSet()
    #alch_system = simulations.generateAlchSystem(system, water.atom_indices)

     # Add reporters to MD simulation.
    traj_reporter = openmm.app.DCDReporter(outfname+'-nc{}.dcd'.format(nstepsNC), opt['trajectory_interval'])
    progress_reporter = openmm.app.StateDataReporter(sys.stdout, separator="\t",
                                reportInterval=opt['reporter_interval'],
                                step=True, totalSteps=opt['nIter']*opt['nstepsMD'],
                                time=True, speed=True, progress=True, remainingTime=True)
    simulations.md.reporters.append(traj_reporter)
    simulations.md.reporters.append(progress_reporter)

    # Run BLUES Simulation
    blues = Simulation(simulations, mover, **opt)
    #blues.runMC(opt['nIter'])
    blues.run(opt['nIter'])


parser = OptionParser()
parser.add_option('-f', '--force', action='store_true', default=False,
                  help='run BLUES example without GPU platform')
parser.add_option('-n','--ncmc', dest='nstepsNC', type='int', default=5000,
                  help='number of NCMC steps')
parser.add_option('-p','--nprop', dest='nprop', type='int', default=3,
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

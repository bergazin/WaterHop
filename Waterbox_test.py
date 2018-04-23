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
from blues.reporters import init_logger, BLUESHDF5Reporter, BLUESStateDataReporter

def runNCMC(platform_name, nstepsNC, nprop, outfname):
    #Generate the ParmEd Structure
    prmtop = '/home/bergazin/WaterHop/water/input_files/wall/oneWat.prmtop'
    inpcrd = '/home/bergazin/WaterHop/water/input_files/wall/oneWat.inpcrd'
    structure = parmed.load_file(prmtop, xyz=inpcrd)
    print('Structure: %s' % structure.topology)

    #Define some options
    opt = { 'temperature' : 500.0, 'friction' : 1, 'dt' : 0.002,
            'nIter' : 100000, 'nstepsNC' : 2500, 'nstepsMD' : 10000, 'nprop' : 5,
            'nonbondedMethod' : 'PME', 'nonbondedCutoff': 10, 'constraints': 'HBonds',
            'trajectory_interval' : 1000, 'reporter_interval' : 1000,
            'ncmc_traj' : None, 'write_move' : False,
            'platform' : platform_name,
            'outfname' : 'testing',
            'verbose' : True}

    logger = init_logger(logging.getLogger(), level=logging.INFO, outfname=opt['outfname'])
    opt['Logger'] = logger
    ## Get the second water in the system, this acts as the "protein" in pure water tests
    # Documentation: http://parmed.github.io/ParmEd/html/amber.html#atom-element-selections
    # http://amber-md.github.io/pytraj/latest/atom_mask_selection.html
    #water_structure = parmed.load_file(prmtop)
    ## To select the protein residues, and not the ligand residues...
    #protein_only_structure = parmed.load_file(prmtop)
    #protein_atoms = protein_only_structure[':GLY,ALA,VAL,LEU,ILE,PRO,PHE,TYR,TRP,SER,THR,CYS,MET,ASN,GLN,LYS,ARG,HIS,ASP,GLU']

    ## Old way
    import mdtraj as md
    wat = md.load('/home/bergazin/WaterHop/water/input_files/wall/oneWat.pdb')
    protein_atoms = wat.topology.select('resid 1916')
    print("protein",protein_atoms)
    
    # Move the coordinates of each of the last waters atoms (the "protein") to the center of the system
    # wat.xyz[0][6518] = np.array([1.80208001,1.62210011,4.75159979])*unit.nanometers
    structure.atoms[6518].xx = [18.0208001,16.2210011,47.5159979]
    structure.atoms[6519].xy = [19.1660004,16.2210011,47.5159979]
    structure.atoms[6520].xz = [17.9679995,18.1479988,47.5159979]
    
    # Define the 'model' object we are perturbing here.
    # Calculate particle masses of object to be moved
    water = WaterTranslationRotationMove(structure, protein_atoms, water_name='WAT')
    water.topology = structure.topology
    water.positions = structure.positions

    # Initialize object that proposes moves.
    mover = MoveEngine(water)
    # Generate the MD, NCMC, ALCHEMICAL Simulation objects
    simulations = SimulationFactory(structure, mover, **opt)
    
    # set the masses of the carbon walls and the last water in the system to 0 to hold it in place
    num_atoms = wat.n_atoms
    for index in range(num_atoms):
        if index < 777 or index > 6517:
            openmm.System.setParticleMass(index=index, mass=0*unit.daltons)
    simulations.createSimulationSet()

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
    #blues.runMC(opt['nIter']) #MC
    blues.run(opt['nIter']) #NCMC


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

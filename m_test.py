import parmed
from simtk import unit
import mdtraj
import numpy as np
import sys, traceback
import math
import copy
import random
import os
from openeye.oechem import *

class Move(object):

    """Move provides methods for calculating properties on the
    object 'move' (i.e ligand) being perturbed in the NCMC simulation.
    This is the base Move class.
    Ex.
        from blues.ncmc import Model
        ligand = Model(structure, 'LIG')
        ligand.calculateProperties()
    Attributes
    ----------
    """

    def __init__(self):
        """Initialize the Move object
        Currently empy.
        """

    def initializeSystem(self, system, integrator):
        """If the system or integrator needs to be modified to perform the move
        (ex. adding a force) this method is called during the start
        of the simulation to change the system or integrator to accomodate that.
        Parameters
        ----------
        system : simtk.openmm.System object
            System to be modified.
        integrator : simtk.openmm.Integrator object
            Integrator to be modified.
        Returns
        -------
        system : simtk.openmm.System object
            The modified System object.
        integrator : simtk.openmm.Integrator object
            The modified Integrator object.
        """
        new_sys = system
        new_int = integrator
        return new_sys, new_int

    def beforeMove(self, context):
        """This method is called at the start of the NCMC portion if the
        context needs to be checked or modified before performing the move
        at the halfway point.
        Parameters
        ----------
        context: simtk.openmm.Context object
            Context containing the positions to be moved.
        Returns
        -------
        context: simtk.openmm.Context object
            The same input context, but whose context were changed by this function.
        """
        return context

    def afterMove(self, context):
        """This method is called at the end of the NCMC portion if the
        context needs to be checked or modified before performing the move
        at the hterTranslationRotation.pyalfway point.
        Parameters
        ----------
        context: simtk.openmm.Context object
            Context containing the positions to be moved.
        Returns
        -------
        context: simtk.openmm.Context object
            The same input context, but whose context were changed by this function.
        """

        return context

    def _error(self, context):
        """This method is called if running during NCMC portion results
        in an error. This allows portions of the context, such as the
        context parameters that would not be fixed by just reverting the
        positions/velocities of the context.
        Parameters
        ----------
        context: simtk.openmm.Context object
            Context containing the positions to be moved.
        Returns
        -------
        context: simtk.openmm.Context object
            The same input context, but whose context were changed by this function.
        """

        return context

    def move(self, context):
        return context


class WaterTranslationRotationMove(Move):
    """ Move that translates a random water within a specified radius of the protein's
    center of mass to another point within that radius, then rotates it around it's center of mass
    Parameters
    ----------
    structure:
        topology: parmed.Topology
            ParmEd topology object containing atoms of the system.
        water_name: str, optional, default='WAT'
            Residue name of the waters in the system.
        radius: float*unit compatible with simtk.unit.nanometers, optional, default=2.0*unit.nanometers
            Defines the radius within the protein center of mass to choose a water
            and the radius in which to randomly translate that water.
    """

    def __init__(self, structure, protein_atoms, water_name='WAT', radius=1.4*unit.nanometers):
        #initialize self attributes
        self.radius = radius
        self.water_name = water_name
        self.water_residues = [] #contains indices of the atoms of the waters
        self.protein_atoms = protein_atoms #contains indices of the atoms in the protein residues
        self.before_ncmc_check = True
        self.traj = mdtraj.load('/home/bergazin/WaterHop/water/input_files/wall/oneWat.pdb')
        print("radius:",radius)

        #go through the topology and identify water and protein residues
        residues = structure.topology.residues()
        #looks for residues with water_name ("WAT")
        for res in residues:
            if res.name == self.water_name: #checks if the name of the residue is 'WAT'
                water_mol = [] #list of each waters atom indices
                for atom in res.atoms():
                   water_mol.append(atom.index) #append the index of each of the atoms of the water residue
                self.water_residues.append(water_mol)#append the water atom indices as a self attribute (above)

        #residues = structure.topology.residues()
        #for res in residues:
        #    if res in ['GLY', 'ALA','VAL','LEU','ILE','PRO','PHE','TYR','TRP','SER','THR','CYS','MET','ASN','GLN','LYS','ARG','HIS','ASP','GLU']:
        #        atom_names = []
       #        atom_index = []
        #        for atom in res.atoms():
        #            atom_names.append(atom.name)
        #            atom_index.append(atom.index)
        #            if 'CA' in atom_names:
        #                self.protein_atoms = self.protein_atoms+atom_index

        #set more self attributes
        #self.atom_indices is used to define the alchemically treated region
        #of the system, in this case the first water in the system
        self.atom_indices = self.water_residues[0] #the atom indices of the first water, this is the alchemical water
        self.topology_protein = structure[self.protein_atoms].topology #typology info based on indices of the atoms in protein residues
        self.topology_water = structure[self.atom_indices].topology  #typology info based on the indices of the first waters atom indices in the system
        self.water_mass = self.getMasses(self.topology_water)#Provides the mass of the specified waters atom
        #print("self.water_mass",self.water_mass)
        self.protein_mass = self.getMasses(self.topology_protein)#Provides the mass of each of the proteins atoms
        #print("self.protein_mass", self.protein_mass)
        self.water_positions = structure[self.atom_indices].positions
        self.protein_positions = structure[self.protein_atoms].positions

    def _random_sphere_point(self, radius):
        """function to generate a uniform random point
        in a sphere of a specified radius.
        Used to randomly translate the water molecule
        Parameters
        ----------
        radius: float
            Defines the radius of the sphere in which a point
            will be uniformly randomly generated.
        """
        r = radius * ( np.random.random()**(1./3.) )  #r (radius) = specified radius * cubed root of a random number between 0.00 and 0.99999
        phi = np.random.uniform(0,2*np.pi) #restriction of phi (or azimuth angle) is set from 0 to 2pi. random.uniform allows the values to be chosen w/ an equal probability
        costheta = np.random.uniform(-1,1) #restriction set from -1 to 1
        theta = np.arccos(costheta) #calculate theta, the angle between r and Z axis
        x = np.sin(theta) * np.cos(phi) #x,y,and z are cartesian coordinates
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        sphere_point = np.array([x, y, z]) * r
        return sphere_point #sphere_point = a random point with an even distribution

    def getMasses(self, topology):
        """Returns a list of masses of the specified ligand atoms.
        Parameters
        ----------
        topology: parmed.Topology
            ParmEd topology object containing atoms of the system.
        """
        print('This is the start of the getMasses function....')
        masses = unit.Quantity(np.zeros([int(topology.getNumAtoms()),1],np.float32), unit.dalton)
        for idx,atom in enumerate(topology.atoms()):
            masses[idx] = atom.element._mass #gets the mass of the atom, adds to list (along with index)
        print('This is the end of the getMasses function....')
        return masses

    def getCenterOfMass(self, positions, masses):
        """Returns the calculated center of mass of the ligand as a np.array
        Parameters
        ----------
        positions: parmed.Structure
            ParmEd positions of the atoms to be moved.
        masses : numpy.array
            np.array of particle masses
        """
        coordinates = np.asarray(positions._value, np.float32) #gives the value of atomic positions as an array
        center_of_mass = parmed.geometry.center_of_mass(coordinates, masses) * positions.unit
        print('This is the end of the getCOM function....')
        return center_of_mass

    def beforeMove(self, nc_context):
        """
        Temporary fterHop/unction (until multiple alchemical regions are supported),
        which is performed at the beginning of a ncmc iteration. Selects
        a random water within self.radius of the protein's center of mass
        and switches the positions and velocities with the alchemical water
        defined by self.atom_indices, effecitvely duplicating mulitple
        alchemical region support.
        Parameters
        ----------
        nc_context: simtk.openmm Context object
            The context which corresponds to the NCMC simulation.
        """

        start_state = nc_context.getState(getPositions=True, getVelocities=True)
        start_pos = start_state.getPositions(asNumpy=True) #gets starting positions
        start_vel = start_state.getVelocities(asNumpy=True) #gets starting velocities
        switch_pos = np.copy(start_pos)*start_pos.unit #starting position (a shallow copy) is * by start_pos.unit to retain units
        switch_vel = np.copy(start_vel)*start_vel.unit #starting vel (a shallow copy) is * by start_pos.unit to retain units
        prot_com = self.getCenterOfMass(switch_pos[self.protein_atoms],
                            masses = self.protein_mass)
        # Choose a random water
        while True:
        #TODO use random.shuffle to pick random particles (limits upper bound)
            water_index = np.random.choice(range(len(self.water_residues)))
            # Exclude the first water in the system (which is the alchemical water) and the last water
            # in the system, whcih is frozen in the center of the box and used for distance calculations
            if 0 < water_index < 1914:
                water_choice = self.water_residues[water_index]
                break

        #replace chosen water's positions/velocities with alchemical water
        for i in range(3):
            #set indices of the alchemical waters atoms equal to the indices of the starting positions of the random waters atoms
            switch_pos[self.atom_indices[i]] = start_pos[water_choice[i]]
            #do the same for velocity
            switch_vel[self.atom_indices[i]] = start_vel[water_choice[i]]
            #set indices of the randomly chosen waters atom equal to alchemical waters atom indices. Same w/ velocity
            switch_pos[water_choice[i]] = start_pos[self.atom_indices[i]]
            switch_vel[water_choice[i]] = start_vel[self.atom_indices[i]]

        print('after_switch', switch_pos[self.atom_indices[0]]) #prints the new indices of the alchemical water
        print('after_switch', switch_pos[self.atom_indices])
        nc_context.setPositions(switch_pos)
        nc_context.setVelocities(switch_vel)
        print("This is the end of the beforeMove function...")
        return nc_context


    def move(self, context):
        """
        This function is called by the blues.MoveEngine object during a simulation.
        Translates the alchemical water randomly within a sphere of self.radius.
        """
        before_move_pos = context.getState(getPositions=True).getPositions(asNumpy=True)
        protein_pos = before_move_pos[self.protein_atoms] #gets the positions from the indices of the atoms in the protein residues in relation to the system
        #find the center of mass and the displacement
        prot_com = self.getCenterOfMass(positions=protein_pos, masses=self.protein_mass) #gets protein COM

        sphere_displacement = self._random_sphere_point(self.radius) #gets a uniform random point in a sphere of a specified radius
        movePos = np.copy(before_move_pos)*before_move_pos.unit #makes a copy of the position of the system from the context

        #Update positions for distance calculation
        self.traj.xyz[0,:,:] = movePos;

        pairs = self.traj.topology.select_pairs(np.array(self.atom_indices[0]).flatten(), np.array(self.protein_atoms[0]).flatten())
        water_distance = mdtraj.compute_distances(self.traj, pairs, periodic=True)
        # Translate water to new location 1.4nm from the center of the system (determined by sphere_displacement)
        for index, resnum in enumerate(self.atom_indices):
            movePos[resnum] = movePos[resnum] - water_distance*movePos.unit + sphere_displacement #new positions of the alch water

        context.setPositions(movePos)
        return context

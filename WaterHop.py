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

    def __init__(self, structure, protein_atoms, water_name='WAT', radius=4.5*unit.nanometers):
        #initialize self attributes
        self.radius = radius #
        self.water_name = water_name
        self.water_residues = [] #contains indices of the atoms of the waters
        self.protein_atoms = protein_atoms #contains indices of the atoms in the protein residues
        #print("self.protein_atoms",self.protein_atoms)
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

        #set more self attributes
        #self.atom_indices is used to define the alchemically treated region
        #of the system, in this case the first water in the system
        self.atom_indices = self.water_residues[0] #the atom indices of the first water, this is the alchemical water
        self.topology_protein = structure[self.protein_atoms].topology
        self.topology_water = structure[self.atom_indices].topology
        self.water_mass = self.getMasses(self.topology_water)
        self.protein_mass = self.getMasses(self.topology_protein)


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
            masses[idx] = atom.element._mass
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

    def beforeMove(self, nc_context):
        """
        Temporary function (until multiple alchemical regions are supported),
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
        prot_com = self.getCenterOfMass(switch_pos[self.protein_atoms], #passes in a copy of the protein atoms starting position
                            masses = self.protein_mass) #passes in list of the proteins atoms masses

        is_inside_sphere = False
        #TODO use random.shuffle to pick random particles (limits upper bound)
        while not is_inside_sphere:
            water_index = np.random.choice(range(len(self.water_residues)))

            #Choose a random water, excluding the first water in the system (the
            #alchemical water) and the very last water in the system (which is
            #frozen in place and used for distance calculations)
            if 0 < water_index < 1914:
                water_choice = self.water_residues[water_index]

            #update the positions from the simulation state before doing distance calculation
            self.traj.xyz[0,:,:] = start_pos;

            #select pairs and compute the distance between the random waters
            #oxygen and the last water in the system via mdtraj
            pairs = self.traj.topology.select_pairs(np.array(water_choice[0]).flatten(), np.array(self.protein_atoms[0]).flatten())
            water_distance = mdtraj.compute_distances(self.traj, pairs, periodic=True)
            water_dist = np.linalg.norm(water_distance)
            #check if the chosen water is less than/equal to the specified radius
            if water_dist <= (self.radius.value_in_unit(unit.nanometers)):
                is_inside_sphere = True
        #replace chosen water's positions/velocities with alchemical water
        for i in range(3):
            switch_pos[self.atom_indices[i]] = start_pos[water_choice[i]]
            switch_vel[self.atom_indices[i]] = start_vel[water_choice[i]]
            switch_pos[water_choice[i]] = start_pos[self.atom_indices[i]]
            switch_vel[water_choice[i]] = start_vel[self.atom_indices[i]]
        print('after_switch', switch_pos[self.atom_indices[0]])
        nc_context.setPositions(switch_pos)
        nc_context.setVelocities(switch_vel)
        return nc_context


    def move(self, context):
        """
        This function is called by the blues.MoveEngine object during a simulation.
        Translates the alchemical water randomly within a sphere of self.radius.
        """
        #get the position of the system from the context
        before_move_pos = context.getState(getPositions=True).getPositions(asNumpy=True)
        protein_pos = before_move_pos[self.protein_atoms]
        prot_com = self.getCenterOfMass(positions=protein_pos, masses=self.protein_mass)
        sphere_displacement = self._random_sphere_point(self.radius)
        movePos = np.copy(before_move_pos)*before_move_pos.unit

        print("movePos[self.atom_indices]",movePos[self.atom_indices])

        #update the positions from the simulation state before doing distance calculation
        self.traj.xyz[0,:,:] = movePos;

        #select pairs and compute the distance between the alchemical waters
        #oxygen and the last water in the system via mdtraj
        pairs = self.traj.topology.select_pairs(np.array(self.atom_indices[0]).flatten(), np.array(self.protein_atoms[0]).flatten())
        water_distance = mdtraj.compute_distances(self.traj, pairs, periodic=True)

        #if the alchemical water is within the radius, translate it
        if np.linalg.norm(water_distance) <= self.radius._value: #see if euc. distance of alch. water is within defined radius
            for index, resnum in enumerate(self.atom_indices):
                print("movePos[resnum]", movePos[resnum])
                movePos[resnum] = movePos[resnum] - water_distance*movePos.unit + sphere_displacement #new positions of the alch water
                print('before', before_move_pos[resnum])
                print('after', movePos[resnum])
            context.setPositions(movePos)
        return context

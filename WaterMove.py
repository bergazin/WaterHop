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

    def __init__(self, structure, protein_atoms, water_name='WAT', radius=1.85*unit.nanometers):
        #initialize self attributes
        self.radius = radius #
        self.water_name = water_name
        self.water_residues = [] #contains indices of the atoms of the waters
        self.protein_atoms = protein_atoms #contains indices of the atoms in the protein residues
        self.before_ncmc_check = True

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
        self.protein_mass = self.getMasses(self.topology_protein)#Provides the mass of each of the proteins atoms
        self.water_positions = structure[self.atom_indices].positions
        self.protein_positions = structure[self.protein_atoms].positions
        #print("self.atom_indices", self.atom_indices)
        print("self.topology_protein", self.topology_protein)
        print("self.topology_water", self.topology_water)
        #print("self.water_mass", self.water_mass)
        #print("self.protein_mass", self.protein_mass)
        print("self.water_residues_all", self.water_residues)
        print("self.water_residues_proW", self.water_residues[1:2])
        print("water_residues_RANGE_LENGTH", range(len(self.water_residues)))
        print("self.water_positions", self.water_positions)
        print("self.protein_positions", self.protein_positions)

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
        masses = unit.Quantity(np.zeros([int(topology.getNumAtoms()),1],np.float32), unit.dalton)
        for idx,atom in enumerate(topology.atoms()):
            masses[idx] = atom.element._mass #gets the mass of the atom, adds to list (along with index)
        print('the masses are', masses)
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
        print('masses', masses)
        print('masses type', type(masses))
        # masses = [element for tupl in masses for element in tupl]
        # print('masses type2', type(masses))
        # print('type2', type(masses[0]))
        # print('masses[0]', masses[0]/ unit.dalton * unit.dalton) #first atoms mass / dalton*dalton, why?
        # print('dir', dir(masses))
        # print('value', positions.value_in_unit(positions.unit))
        # print(positions) #positons: A list of 3-element Quantity tuples of dimension length representing the atomic positions for every atom in the system.
        coordinates = np.asarray(positions._value, np.float32) #gives the value of atomic positions as an array
        center_of_mass = parmed.geometry.center_of_mass(coordinates, masses) * positions.unit
        return center_of_mass

   def beforeMove(self, nc_context):
        """
        Temporary fterHop/unction (until multiple alchemical regions are supported), which is performed at the beginning of a 
        ncmc iteration. Selects a random water within self.radius of the protein's center of mass and switches the positions 
        and velocities with the alchemical water defined by self.atom_indices, effecitvely duplicating mulitple alchemical region support.
        Parameters
        ----------
        nc_context: simtk.openmm Context object
            The context which corresponds to the NCMC simulation.
        """
        start_state = nc_context.getState(getPositions=True, getVelocities=True)
        start_pos = start_state.getPositions(asNumpy=True) #gets starting positions
        print('start_pos', start_pos[self.atom_indices[0]]) #prints starting position of the first water atom

        start_vel = start_state.getVelocities(asNumpy=True) #gets starting velocities
        switch_pos = np.copy(start_pos)*start_pos.unit #starting position (a shallow copy) is * by start_pos.unit to retain units
        switch_vel = np.copy(start_vel)*start_vel.unit #starting vel (a shallow copy) is * by start_pos.unit to retain units
        print('switch_pos', switch_pos)

        prot_com = self.getCenterOfMass(switch_pos[self.protein_atoms], #passes in a copy of the protein atoms starting position
                            masses = self.protein_mass) #passes in list of the proteins atoms masses
        is_inside_sphere = False
        #TODO use random.shuffle to pick random particles (limits upper bound)
        while not is_inside_sphere:
            #water_choice = np.random.choice(water_residues)
            water_index = np.random.choice(range(len(self.water_residues)))
            if water_index > 1: #Don't want to switch alch. water vel/pos with itself or the 2nd water in the system (which is acting as protein)
                water_choice = self.water_residues[water_index]
            #print('water_choice', water_choice)
            import mdtraj
            #oxygen_pos = start_pos[water_choice[0]] #Gets XYZ coords of oxygen
            oxygen_pos = start_pos[water_choice[0]]
            #print("oxygen_pos",oxygen_pos)
            protein_choice = self.water_residues[1]
            protein_pos = start_pos[protein_choice[0]] #proteins
            #print("protein_pos", protein_pos)
            traj = mdtraj.load('/home/bergazin/WaterHop/water/input_files/onlyWaterBox/BOX1.pdb')
            pairs = traj.topology.select_pairs(oxygen_pos._value, protein_pos._value)
            #print("pairs", pairs)
            water_distance = mdtraj.compute_distances(traj, pairs, periodic=True)
            #print("water_distance", water_distance)
            water_dist = np.linalg.norm(water_distance)
            if water_dist <= (self.radius.value_in_unit(unit.nanometers)):
                is_inside_sphere = True

        #replace chosen water's positions/velocities with alchemical water
        for i in range(3):
            #set indices of the alchemical waters atoms equal to the indices of the starting positions of the random waters atoms
            switch_pos[self.atom_indices[i]] = start_pos[water_choice[i]]
            #do the same for velocity
            switch_vel[self.atom_indices[i]] = start_vel[water_choice[i]]
            #set indices of the randomly chosen waters atom equal to alchemical waters atom indices. Same w/ velocity
            switch_pos[water_choice[i]] = start_pos[self.atom_indices[i]]
            switch_vel[water_choice[i]] = start_vel[self.atom_indices[i]]
            print("Velocities and positions have been switched")

        print('after_switch', switch_pos[self.atom_indices[0]]) #prints the new indices of the alchemical water
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
        protein_pos = before_move_pos[self.protein_atoms] #gets the positions from the indices of the atoms in the protein residues in relation to the system

        #find the center of mass and the displacement
        prot_com = self.getCenterOfMass(positions=protein_pos, masses=self.protein_mass) #gets protein COM
        sphere_displacement = self._random_sphere_point(self.radius) #gets a uniform random point in a sphere of a specified radius
        movePos = np.copy(before_move_pos)*before_move_pos.unit #makes a copy of the position of the system from the context
        print('movePos LOOK HEREEEEEEEEEE', movePos[self.atom_indices]) #gets positions of the alchemical waters atoms from the context
        print('center of mass', prot_com) #prints the protein COM
        print('COM_TYPE', type(prot_com._value))
        print('Water coord', self.atom_indices) #prints alchemical waters atoms indices

        #first atom in the water molecule (which is Oxygen) was used to measure the distance
        #water_displacement = movePos[self.atom_indices[0]] - prot_com #estimate distance of the water from the proteins com.
        oxygen_pos = movePos[self.atom_indices[0]]
        protein_choice = self.water_residues[1]
        prot_pos = movePos[protein_choice[0]]
        
        traj = mdtraj.load('/home/bergazin/WaterHop/water/input_files/onlyWaterBox/BOX1.pdb')
        pairs = traj.topology.select_pairs(oxygen_pos._value, prot_pos._value)
        water_distance = mdtraj.compute_distances(traj, pairs, periodic=True)
        
        #if the alchemical water is within the radius, translate it
        if np.linalg.norm(water_distance) <= self.radius._value: #see if euc. distance of alch. water is within defined radius
            for index, resnum in enumerate(self.atom_indices):
                print("movePos[resnum]", movePos[resnum])
                movePos[resnum] = movePos[resnum] - water_displacement + sphere_displacement #new positions of the alch water
                print('before', before_move_pos[resnum])
                print('after', movePos[resnum])
            context.setPositions(movePos)
        return context
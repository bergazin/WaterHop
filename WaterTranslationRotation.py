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

    def __init__(self, structure, protein_atoms, water_name='WAT', radius=5.0*unit.nanometers):
        #initialize self attributes
        self.radius = radius 
        self.water_name = water_name
        self.water_residues = [] 
        self.protein_atoms = protein_atoms 
        self.before_ncmc_check = True
        self.traj = mdtraj.load('/home/bergazin/WaterHop/water/input_files/wall/oneWat.pdb')

        # Go through the topology and identify water and protein residues
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

        #Set more self attributes
        #self.atom_indices is used to define the alchemically treated region
        #of the system, in this case the first water in the system
        self.atom_indices = self.water_residues[0] 
        self.topology_protein = structure[self.protein_atoms].topology 
        self.topology_water = structure[self.atom_indices].topology  
        self.water_mass = self.getMasses(self.topology_water)
        self.protein_mass = self.getMasses(self.topology_protein)
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
        r = radius * ( np.random.random()**(1./3.) )  
        phi = np.random.uniform(0,2*np.pi) 
        costheta = np.random.uniform(-1,1) 
        theta = np.arccos(costheta) 
        x = np.sin(theta) * np.cos(phi) 
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        sphere_point = np.array([x, y, z]) * r
        return sphere_point 

    def getMasses(self, topology):
        """Returns a list of masses of the specified ligand atoms.
        Parameters
        ----------
        topology: parmed.Topology
            ParmEd topology object containing atoms of the system.
        """
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
        coordinates = np.asarray(positions._value, np.float32) 
        center_of_mass = parmed.geometry.center_of_mass(coordinates, masses) * positions.unit
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
        start_pos = start_state.getPositions(asNumpy=True) 
        start_vel = start_state.getVelocities(asNumpy=True) 
        switch_pos = np.copy(start_pos)*start_pos.unit 
        switch_vel = np.copy(start_vel)*start_vel.unit 
        prot_com = self.getCenterOfMass(switch_pos[self.protein_atoms], masses = self.protein_mass) 
        
        is_inside_sphere = False
        #TODO use random.shuffle to pick random particles (limits upper bound)
        while not is_inside_sphere:
            water_index = np.random.choice(range(len(self.water_residues)))

            if 0 < water_index < 1914: #Don't want to switch alch. water vel/pos with itself or the 2nd water in the system (which is acting as protein)
                water_choice = self.water_residues[water_index]
                
            #Select alch. waters oxygen and the "protein"/waters oxygen to be used for distance calculations (ie to see if it's in radius)
            oxygen_pos1 = np.array(water_choice[0])
            oxygen_pos = oxygen_pos1.flatten()

            protein_indice = self.protein_atoms
            protein_choice1 = np.array(self.protein_atoms[0])
            protein_choice = protein_choice1.flatten()
            
            #Update the positions from the simulation state.
            self.traj.xyz[0,:,:] = start_pos;
            
            #Compute distances
            pairs = self.traj.topology.select_pairs(oxygen_pos, protein_choice)
            water_distance = mdtraj.compute_distances(self.traj, pairs, periodic=True)
            water_dist = np.linalg.norm(water_distance)
            # Check if the randomly chosen water is within the radius
            if water_dist <= (self.radius.value_in_unit(unit.nanometers)):
                is_inside_sphere = True

        #replace chosen water's positions/velocities with alchemical water
        for i in range(3):
            switch_pos[self.atom_indices[i]] = start_pos[water_choice[i]]
            switch_vel[self.atom_indices[i]] = start_vel[water_choice[i]]
            switch_pos[water_choice[i]] = start_pos[self.atom_indices[i]]
            switch_vel[water_choice[i]] = start_vel[self.atom_indices[i]]

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
        
        #Select alch. waters oxygen and the "protein"/waters oxygen to be used for distance calculations (ie to see if it's in radius)
        oxygen_pos1 = np.array(self.atom_indices[0])
        oxygen_pos = oxygen_pos1.flatten()
        protein_indices = self.protein_atoms
        protein_choice1 = np.array(protein_indices[0])
        protein_choice = protein_choice1.flatten()

        #Update the positions from the simulation state.
        self.traj.xyz[0,:,:] = movePos;
        
        #Compute distances
        pairs = self.traj.topology.select_pairs(protein_choice, oxygen_pos)
        water_distance = mdtraj.compute_distances(self.traj, pairs, periodic=True)

        #if the alchemical water is within the radius, translate it
        if np.linalg.norm(water_distance) <= self.radius._value:
            for index, resnum in enumerate(self.atom_indices):
                movePos[resnum] = movePos[resnum] - water_distance*movePos.unit + sphere_displacement 
            context.setPositions(movePos)
        return context

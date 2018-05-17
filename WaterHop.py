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

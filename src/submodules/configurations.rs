use crate::submodules::boundary_conditions::*;
use crate::submodules::type_lib::NumericData;
use crate::submodules::type_lib::Position;
use ndarray::array;
use ndarray::Array1;
use ndarray::Array2;
use rand::random;

use super::energy_evaluation::Dimer;
use super::energy_evaluation::MagneticDimer;
use super::energy_evaluation::PotentialKinds;
use super::energy_evaluation::PotentialVariableKinds;
use super::ensembles::EnsembleTrait;
use super::mc_state::MCState;

#[derive(Clone)]
pub struct Configuration<BC: BoundaryConditionTrait> {
    pub position_vector: Array1<Position>,
    pub boundary_condition: BC,
    pub number_of_atoms: usize,
}

impl<BC: BoundaryConditionTrait> Configuration<BC> {
    pub fn new_spherical(position_vector: Array1<Position>, boundary_condition: Spherical) -> Configuration<Spherical> {
        let number_of_atoms = position_vector.len();
        Configuration {
            position_vector,
            boundary_condition: boundary_condition.clone(),
            number_of_atoms,
        }
    }

    pub fn new_cubic(position_vector: Array1<Position>, boundary_condition: Cubic) -> Configuration<Periodic> {
        let number_of_atoms = position_vector.len();
        Configuration {
            position_vector,
            boundary_condition: Periodic::Cubic(boundary_condition.clone()),
            number_of_atoms,
        }
    }

    pub fn new_rhombic(position_vector: Array1<Position>, boundary_condition: Rhombic) -> Configuration<Periodic> {
        let number_of_atoms = position_vector.len();
        Configuration {
            position_vector,
            boundary_condition: Periodic::Rhombic(boundary_condition.clone()),
            number_of_atoms,
        }
    }

    pub fn get_centre(&self) -> Position {
        let mut centre = [0.0; 3];
        for i in 0..3 {
            for j in 0..self.number_of_atoms {
                centre[i] += self.position_vector[j][i];
            }
            centre[i] /= self.number_of_atoms as NumericData;
        }
        centre
    }

    pub fn recentre(&mut self) {
        let centre = self.get_centre();
        for i in 0..self.number_of_atoms {
            for j in 0..3 {
                self.position_vector[i][j] -= centre[j];
            }
        }
    }

    pub fn get_distance2_mat(&self) -> Array2<NumericData> {
        let mut distance2_mat = Array2::<NumericData>::zeros((self.number_of_atoms, self.number_of_atoms));
        for i in 0..self.number_of_atoms {
            for j in 0..i {
                distance2_mat[[i, j]] = self.boundary_condition.distance_squared(&self.position_vector[i], &self.position_vector[j]);
                distance2_mat[[j, i]] = distance2_mat[[i, j]];
            }
        }
        distance2_mat
    }

    pub fn get_tan_mat(&self) -> Array2<NumericData> {
        let mut tan_mat = Array2::<NumericData>::zeros((self.number_of_atoms, self.number_of_atoms));
        for i in 0..self.number_of_atoms {
            for j in 0..i {
                tan_mat[[i, j]] = self.boundary_condition.get_tan(&self.position_vector[i], &self.position_vector[j]);
                tan_mat[[j, i]] = tan_mat[[i, j]];
            }
        }
        tan_mat
    }
}

impl Configuration<Periodic> {
    pub fn volume_change(&mut self, max_length: NumericData, max_v_change: NumericData) -> NumericData {
        match self.boundary_condition {
            Periodic::Cubic(ref mut cubic) => {
                let mut scale = ((random::<NumericData>()-0.5)*max_v_change).exp().cbrt();
                if cubic.side_length >= max_length && scale > 1.0 {
                    scale = 1.0;
                }
                self.position_vector = self.position_vector.iter().map(|pos| [pos[0]*scale, pos[1]*scale, pos[2]*scale]).collect();
                *cubic = Cubic{ side_length: cubic.side_length*scale };
                scale
            },
            Periodic::Rhombic(ref mut rhombic) => {
                let mut scale = ((random::<NumericData>()-0.5)*max_v_change).exp().cbrt();
                if rhombic.side_length >= max_length && scale > 1.0 {
                    scale = 1.0;
                }
                self.position_vector = self.position_vector.iter().map(|pos| [pos[0]*scale, pos[1]*scale, pos[2]*scale]).collect();
                *rhombic = Rhombic { side_length: rhombic.side_length*scale, side_height: rhombic.side_height*scale };
                scale
            },
        }
    }

    pub fn get_mc_state<E: EnsembleTrait>(&self, temperature: NumericData, beta: NumericData, ensemble: &E, potential_enum: &PotentialKinds) -> MCState<Periodic> {
        let distance2_mat = self.get_distance2_mat();
        let n_atoms = self.number_of_atoms;
        let max_boxlength = self.boundary_condition.max_length();
        let ensemble_variables = ensemble.set_ensemble_variables(Some(&self));

        let potential_variables;
        let energy;
        match potential_enum {
            PotentialKinds::ELJ(pot) => {
                let mut potvars = pot.set_variables(self.number_of_atoms);
                energy = pot.initialise_energy_periodic(&self, &distance2_mat, &mut potvars, self.boundary_condition.get_r_cut());
                potential_variables = PotentialVariableKinds::Dimer(potvars);
            },
            PotentialKinds::ELJEven(pot) => {
                let mut potvars = pot.set_variables(self.number_of_atoms);
                energy = pot.initialise_energy_periodic(&self, &distance2_mat, &mut potvars, self.boundary_condition.get_r_cut());
                potential_variables = PotentialVariableKinds::Dimer(potvars);
                
            },
            PotentialKinds::ELJB(pot) => {
                let mut potvars = pot.set_variables(&self);
                energy = pot.initialise_energy_periodic(&self, &distance2_mat, &mut potvars, self.boundary_condition.get_r_cut());
                potential_variables = PotentialVariableKinds::Magnetic(potvars);
            },
        }

        MCState {
            temperature,
            beta,
            configuration: self.clone(),
            distance2_mat,
            new_distance_vec: Array1::<NumericData>::zeros(n_atoms),
            new_energy: 0.0,
            total_energy: energy,
            potential_variables,
            ensemble_variables,
            hamiltonian: (0.0, 0.0),
            max_displacement: vec![0.1,0.1,1.],
            max_boxlength,
            count_atom: [0; 2],
            count_volume: [0; 2],
            count_exc: [0; 2],
        }
    }
}

impl Configuration<Spherical> {
    pub fn get_mc_state<E: EnsembleTrait>(&self, temperature: NumericData, beta: NumericData, ensemble: &E, potential_enum: &PotentialKinds) -> MCState<Spherical> {
        let distance2_mat = self.get_distance2_mat();
        let n_atoms = self.number_of_atoms;
        let max_boxlength = self.boundary_condition.max_length();
        let ensemble_variables = ensemble.set_ensemble_variables(None);

        let potential_variables;
        let energy;
        match potential_enum {
            PotentialKinds::ELJ(pot) => {
                let mut potvars = pot.set_variables(self.number_of_atoms);
                energy = pot.initialise_energy_aperiodic(&distance2_mat, &mut potvars);
                potential_variables = PotentialVariableKinds::Dimer(potvars);
            },
            PotentialKinds::ELJEven(pot) => {
                let mut potvars = pot.set_variables(self.number_of_atoms);
                energy = pot.initialise_energy_aperiodic(&distance2_mat, &mut potvars);
                potential_variables = PotentialVariableKinds::Dimer(potvars);
                
            },
            PotentialKinds::ELJB(pot) => {
                let mut potvars = pot.set_variables(&self);
                energy = pot.initialise_energy_aperiodic(&distance2_mat, &mut potvars);
                potential_variables = PotentialVariableKinds::Magnetic(potvars);
            },
        }

        MCState {
            temperature,
            beta,
            configuration: self.clone(),
            distance2_mat,
            new_distance_vec: Array1::<NumericData>::zeros(n_atoms),
            new_energy: 0.0,
            total_energy: energy,
            potential_variables,
            ensemble_variables,
            hamiltonian: (0.0, 0.0),
            max_displacement: vec![0.1,0.1,1.],
            max_boxlength,
            count_atom: [0; 2],
            count_volume: [0; 2],
            count_exc: [0; 2],
        }
    }
}
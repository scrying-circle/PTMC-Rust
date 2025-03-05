use std::mem::swap;

use ndarray::{Array1, Array2};
use super::{boundary_conditions::{AperiodicTrait, BoundaryConditionTrait, Cubic, Periodic, PeriodicTrait, Rhombic, Spherical}, configurations::Configuration, energy_evaluation::{Dimer, MagneticDimer, PotentialKinds, PotentialVariableKinds, PotentialVariablesTrait}, ensembles::{EnsembleTrait, EnsembleVariableKinds, EnsembleVariablesTrait, MoveStrategy, MoveType, NPTVariables, NPT}, input_params::Output, type_lib::{NumericData, Position}};

pub struct MCState<BC: BoundaryConditionTrait> {
    pub temperature: NumericData,
    pub beta: NumericData,
    pub configuration: Configuration<BC>,
    pub distance2_mat: Array2<NumericData>,
    pub new_distance_vec: Array1<NumericData>,
    pub new_energy: NumericData,
    pub total_energy: NumericData,
    pub potential_variables: PotentialVariableKinds,
    pub ensemble_variables: EnsembleVariableKinds,
    pub hamiltonian: (NumericData, NumericData),
    pub max_displacement: Vec<NumericData>,
    pub max_boxlength: NumericData,
    pub count_atom: [usize; 2],
    pub count_volume: [usize; 2],
    pub count_exc: [usize; 2],
}

impl MCState<Periodic> {
    pub fn volume_change(&mut self) {
        match self.ensemble_variables {
            EnsembleVariableKinds::NPT(ref mut npt) => {
                let scale = self.configuration.volume_change(self.max_boxlength, self.max_displacement[1]);
                npt.new_r_cut = npt.trial_config.boundary_condition.get_r_cut();
                npt.new_dist2_mat = npt.trial_config.get_distance2_mat() * scale;
            },
            _ => panic!("Wrong ensemble for volume change.")
        }
    }

    pub fn atom_displacement(&mut self) {
        self.ensemble_variables.set_trial_move(self.configuration.boundary_condition.atom_displacement(&self.configuration.position_vector[self.ensemble_variables.atom_index()], self.max_displacement[0]));
        for (i, b) in self.configuration.position_vector.iter().enumerate() {
            self.new_distance_vec[i] = self.configuration.boundary_condition.distance_squared(&self.ensemble_variables.trial_move(), b);
        }
        self.new_distance_vec[self.ensemble_variables.atom_index()] = 0.0;
    }

    pub fn generate_move(&mut self, move_type: &MoveType) {
        match move_type {
            MoveType::AtomMove => self.atom_displacement(),
            MoveType::VolumeMove => self.volume_change(),
            _ => panic!("Wrong move type for periodic boundary condition.")
        }
    }

    pub fn get_energy(&mut self, potential: &PotentialKinds, move_type: &MoveType) {
        match move_type {
            MoveType::AtomMove => {
                match potential {
                    PotentialKinds::ELJ(dimer) => {
                        self.new_energy = self.total_energy + dimer.energy_update(self.ensemble_variables.atom_index(), &self.distance2_mat, &self.new_distance_vec)
                    },
                    PotentialKinds::ELJEven(dimer) => {
                        self.new_energy = self.total_energy + dimer.energy_update(self.ensemble_variables.atom_index(), &self.distance2_mat, &self.new_distance_vec)
                    },
                    PotentialKinds::ELJB(magnetic_dimer) => {
                        match self.potential_variables {
                            PotentialVariableKinds::Magnetic(ref mut magnetic) => {
                                self.new_energy = self.total_energy + magnetic_dimer.energy_update(self.ensemble_variables.atom_index(), &self.distance2_mat, &self.new_distance_vec, magnetic, self.ensemble_variables.trial_move(), &self.configuration)
                            },
                            _ => panic!("Wrong potential variables for magnetic dimer.")
                        }
                    },
                }
            },
            MoveType::VolumeMove => {
                if let EnsembleVariableKinds::NPT(npt) = &self.ensemble_variables {
                    match potential {
                        PotentialKinds::ELJ(dimer) => {
                            self.new_energy = self.total_energy + dimer.energy_update(self.ensemble_variables.atom_index(), &npt.new_dist2_mat, &self.new_distance_vec)
                        },
                        PotentialKinds::ELJEven(dimer) => {
                            self.new_energy = self.total_energy + dimer.energy_update(self.ensemble_variables.atom_index(), &npt.new_dist2_mat, &self.new_distance_vec)
                        },
                        PotentialKinds::ELJB(magnetic_dimer) => {
                            match self.potential_variables {
                                PotentialVariableKinds::Magnetic(ref mut magnetic) => {
                                    self.new_energy = self.total_energy + magnetic_dimer.energy_update(self.ensemble_variables.atom_index(), &npt.new_dist2_mat, &self.new_distance_vec, magnetic, self.ensemble_variables.trial_move(), &npt.trial_config)
                                },
                                _ => panic!("Wrong potential variables for magnetic dimer.")
                            }
                        },
                    }
                }
            },
            _ => panic!("Wrong move type for periodic boundary condition.")
        }
    }

    pub fn metropolis_condition(&self, move_type: &MoveType, ensemble: &NPT) -> bool {
        match move_type {
            MoveType::AtomMove | MoveType::AtomSwap => {
                let delta_energy = self.new_energy - self.total_energy;
                let probability = (-self.beta * delta_energy).exp();
                let random_number = rand::random::<NumericData>();
                random_number < probability
            },
            MoveType::VolumeMove => {
                let final_volume = match self.ensemble_variables {
                    EnsembleVariableKinds::NPT(ref npt) => npt.trial_config.boundary_condition.get_volume(),
                    _ => panic!("Wrong ensemble for volume move.")
                };
                let initial_volume = self.configuration.boundary_condition.get_volume();
                let delta_energy = self.new_energy - self.total_energy;
                let delta_h = delta_energy + ensemble.pressure * (final_volume - initial_volume);
                let probability = (-self.beta * delta_h + ensemble.n_atoms as NumericData *(final_volume/initial_volume).ln()).exp();
                let random_number = rand::random::<NumericData>();
                random_number < probability
            }
        }
    }

    pub fn update_max_stepsize(&mut self, n_update: usize, min_acceptance: NumericData, max_acceptance: NumericData, ensemble: &NPT) {
        let acc_rate = (self.count_atom[1] / (n_update * ensemble.n_atom_moves)) as NumericData;
        if acc_rate < min_acceptance {
            self.max_displacement[0] *= 0.9;
        } else if acc_rate > max_acceptance {
            self.max_displacement[0] *= 1.1;
        }
        self.count_atom[1] = 0;

        let acc_rate = (self.count_volume[1] / (n_update * ensemble.n_volume_moves)) as NumericData;
        if acc_rate < min_acceptance {
            self.max_displacement[1] *= 0.9;
        } else if acc_rate > max_acceptance {
            self.max_displacement[1] *= 1.1;
        }
        self.count_volume[1] = 0;
    }

    pub fn swap_config_v(&mut self) {
        if let EnsembleVariableKinds::NPT(ref mut nptvars) = self.ensemble_variables {
            match &nptvars.trial_config.boundary_condition {
                Periodic::Cubic(Cubic {side_length}) => {
                    self.configuration = Configuration::<Periodic>::new_cubic(nptvars.trial_config.position_vector.clone(), Cubic{side_length: *side_length});
                },
                Periodic::Rhombic(Rhombic {side_length, side_height}) => {
                    self.configuration = Configuration::<Periodic>::new_rhombic(nptvars.trial_config.position_vector.clone(), Rhombic{side_length: *side_length, side_height: *side_height});
                }
            }

            self.distance2_mat = nptvars.new_dist2_mat.clone();
            //self.potential_variables.set_en_atom_vec(); bug in original code
            self.total_energy = self.new_energy;
            self.count_volume[0] += 1;
            self.count_volume[1] += 1;

            nptvars.r_cut = nptvars.new_r_cut;
        }
    }

    pub fn acc_test(&mut self, move_type: &MoveType, ensemble: &NPT) {
        if self.metropolis_condition(move_type, ensemble) {
            self.swap_config(move_type);
        }
    }

    pub fn swap_config(&mut self, move_type: &MoveType) {
        match move_type {
            MoveType::AtomMove => {
                self.swap_atom_config();
            },
            _ => panic!("not implemenented yet rip")
        }
    }
    
    pub fn mc_move(&mut self, move_strategy: &MoveStrategy<NPT>, potential: &PotentialKinds) {
        self.ensemble_variables.set_atom_index(rand::random_range(1..move_strategy.len()));
        let move_type = &move_strategy.move_type_vec[self.ensemble_variables.atom_index()];
        self.generate_move(move_type);
        self.get_energy(potential, move_type);
        self.acc_test(move_type, &move_strategy.ensemble);
    }
}

impl MCState<Spherical> {
    pub fn atom_displacement(&mut self) {
        let mut count: usize = 0;
        let mut trial_pos = self.configuration.boundary_condition.atom_displacement(&self.configuration.position_vector[self.ensemble_variables.atom_index()], self.max_displacement[0]);
        loop {
            if !self.configuration.boundary_condition.check_boundary(&trial_pos) {
                break;
            }
            count += 1;
            if count == 50 {
                self.configuration.recentre();
            } else {
                trial_pos = self.configuration.boundary_condition.atom_displacement(&self.configuration.position_vector[self.ensemble_variables.atom_index()], self.max_displacement[0]);
                if count == 100 {panic!("Too many moves out of binding sphere.")}
            }
        }
    }

    pub fn swap_atoms(&mut self) {
        match self.ensemble_variables {
            EnsembleVariableKinds::NNVT(ref mut nnvt)  => {
                let index1 = rand::random_range(1..nnvt.n1n2.0);
                let index2 = rand::random_range(nnvt.n1n2.0+1..nnvt.n1n2.1);
                nnvt.swap_indices = (index1, index2);
            },
            _ => panic!("Wrong ensemble for atom swap.")
        }
    }

    pub fn generate_move(&mut self, move_type: &MoveType) {
        match move_type {
            MoveType::AtomMove => self.atom_displacement(),
            MoveType::AtomSwap => self.swap_atoms(),
            _ => panic!("Wrong move type for aperiodic boundary condition.")
        }
    }

    pub fn get_energy(&mut self, potential: &PotentialKinds, move_type: &MoveType) {
        match self.ensemble_variables {
            EnsembleVariableKinds::NVT(_) => {
                if let MoveType::AtomMove = move_type {
                    match potential {
                        PotentialKinds::ELJ(dimer) => {
                            self.new_energy = self.total_energy + dimer.energy_update(self.ensemble_variables.atom_index(), &self.distance2_mat, &self.new_distance_vec)
                        },
                        PotentialKinds::ELJEven(dimer) => {
                            self.new_energy = self.total_energy + dimer.energy_update(self.ensemble_variables.atom_index(), &self.distance2_mat, &self.new_distance_vec)
                        },
                        PotentialKinds::ELJB(magnetic_dimer) => {
                            match self.potential_variables {
                                PotentialVariableKinds::Magnetic(ref mut magnetic) => {
                                    self.new_energy = self.total_energy + magnetic_dimer.energy_update(self.ensemble_variables.atom_index(), &self.distance2_mat, &self.new_distance_vec, magnetic, self.ensemble_variables.trial_move(), &self.configuration)
                                },
                                _ => panic!("Wrong potential variables for magnetic dimer.")
                            }
                        },
                    }
                }
            },
            EnsembleVariableKinds::NNVT(_) => {
                match move_type {
                    MoveType::AtomMove => {
                        match potential {
                            PotentialKinds::ELJ(dimer) => {
                                self.new_energy = self.total_energy + dimer.energy_update(self.ensemble_variables.atom_index(), &self.distance2_mat, &self.new_distance_vec)
                            },
                            PotentialKinds::ELJEven(dimer) => {
                                self.new_energy = self.total_energy + dimer.energy_update(self.ensemble_variables.atom_index(), &self.distance2_mat, &self.new_distance_vec)
                            },
                            PotentialKinds::ELJB(magnetic_dimer) => {
                                match self.potential_variables {
                                    PotentialVariableKinds::Magnetic(ref mut magnetic) => {
                                        self.new_energy = self.total_energy + magnetic_dimer.energy_update(self.ensemble_variables.atom_index(), &self.distance2_mat, &self.new_distance_vec, magnetic, self.ensemble_variables.trial_move(), &self.configuration)
                                    },
                                    _ => panic!("Wrong potential variables for magnetic dimer.")
                                }
                            },
                        }
                    },
                    MoveType::AtomSwap => {
                        ()
                    },
                    _ => panic!("Wrong move type for aperiodic boundary condition.")
                }
            },
            _ => panic!("Wrong ensemble for aperiodic boundary condition.")
        }
    }

    pub fn metropolis_condition(&self) -> bool {
        let delta_energy = self.new_energy - self.total_energy;
        let probability = (-self.beta * delta_energy).exp();
        let random_number = rand::random::<NumericData>();
        random_number < probability
    }

    pub fn update_max_stepsize<E: EnsembleTrait>(&mut self, n_update: usize, min_acceptance: NumericData, max_acceptance: NumericData, ensemble: &E) {
        let acc_rate = (self.count_atom[1] / (n_update * ensemble.get_n_atom_moves())) as NumericData;
        if acc_rate < min_acceptance {
            self.max_displacement[0] *= 0.9;
        } else if acc_rate > max_acceptance {
            self.max_displacement[0] *= 1.1;
        }
        self.count_atom[1] = 0;
    }

    pub fn acc_test<E: EnsembleTrait>(&mut self, move_type: &MoveType) {
        if self.metropolis_condition() {
            self.swap_config(move_type);
        }
    }

    pub fn swap_config(&mut self, move_type: &MoveType) {
        match move_type {
            MoveType::AtomMove => {
                self.swap_atom_config();
            },
            _ => panic!("not implemenented yet rip")
        }
    }

    pub fn mc_move<E: EnsembleTrait>(&mut self, move_strategy: &MoveStrategy<E>, potential: &PotentialKinds) {
        self.ensemble_variables.set_atom_index(rand::random_range(1..move_strategy.len()));
        let move_type = &move_strategy.move_type_vec[self.ensemble_variables.atom_index()];
        self.generate_move(move_type);
        self.get_energy(potential, move_type);
        self.acc_test::<E>(move_type);
    }
}

impl<BC: BoundaryConditionTrait> MCState<BC> {
    pub fn swap_atom_config(&mut self) {
        let atom_index = self.ensemble_variables.atom_index();
        let trial_pos = self.ensemble_variables.trial_move();
        self.configuration.position_vector[atom_index] = *trial_pos;
        self.distance2_mat.row_mut(atom_index).assign(&self.new_distance_vec);
        self.distance2_mat.column_mut(atom_index).assign(&self.new_distance_vec);
        swap(&mut self.new_energy, &mut self.total_energy);
        self.count_atom[0] += 1;
        self.count_atom[1] += 1;

        self.potential_variables.swap_vars(atom_index);
    }

    pub fn reset_counters(&mut self) {
        self.count_atom = [0, 0];
        self.count_volume = [0, 0];
        self.count_exc = [0, 0];
    }
}
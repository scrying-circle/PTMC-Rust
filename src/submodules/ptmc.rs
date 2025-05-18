use std::mem::swap;
use rayon::prelude::*;
use crate::submodules::system_solver::SystemSolver;

use super::{boundary_conditions::{BoundaryConditionTrait, Cubic, Periodic, PeriodicTrait, Rhombic, Spherical}, configurations::Configuration, energy_evaluation::PotentialKinds, ensembles::{EnsembleTrait, MoveStrategy, NPT}, initialisation::Initialiser, input_params::{EnergyBounds, MCParams, Output, TempGrid}, mc_state::MCState, test_contents::TestContents, type_lib::NumericData};

pub struct PTMC<E: EnsembleTrait + Sync, BC: BoundaryConditionTrait> {
    pub temp_grid: TempGrid,
    pub mc_params: MCParams,
    pub ensemble: E,
    pub move_strategy: MoveStrategy<E>,
    pub mc_states: Vec<MCState<BC>>,
    pub potential: PotentialKinds,
    pub results: Output,
    pub energy_bounds: EnergyBounds,
    pub start_counter: usize,
    pub n_steps: usize,
}

impl<E: EnsembleTrait + Sync, BC: BoundaryConditionTrait> PTMC<E, BC> {
    pub fn exchange_acceptance(beta1: NumericData, beta2: NumericData, energy1: NumericData, energy2: NumericData) -> NumericData {
        ((beta1 - beta2) * (energy1 - energy2)).exp().min(1.0)
    }

    pub fn exchange_trajectories(&mut self, index: usize) {
        let (left, right) = self.mc_states.split_at_mut(index+1);
        let state1 = left.last_mut().unwrap();
        let state2 = right.first_mut().unwrap();
        swap(&mut state1.configuration, &mut state2.configuration);
        swap(&mut state1.distance2_mat, &mut state2.distance2_mat);
        swap(&mut state1.total_energy, &mut state2.total_energy);
        swap(&mut state1.potential_variables, &mut state2.potential_variables);
        swap(&mut state1.ensemble_variables, &mut state2.ensemble_variables);
    }

    pub fn postprocess(&mut self) -> SystemSolver {
        let energy_vector = self.results.readfile();
        TestContents::print(&self.results.energy_histogram, "raw_hist.json");
        TestContents::print(&self.temp_grid.beta_grid, "beta.json");
        TestContents::print(&energy_vector, "energy_vector.json");
        let (processed_histogram, processed_energy_vector, sums) = self.results.process_histogram(&energy_vector);
        TestContents::print(&processed_histogram, "processed_histogram.json");
        TestContents::print(&self.results.energy_histogram.shape()[0], "n_traj.json");
        TestContents::print(&self.results.energy_histogram.shape()[1], "n_bins.json");
        SystemSolver::multihistogram(&processed_histogram, &processed_energy_vector, &self.temp_grid.beta_grid, &sums)
    }
}

impl <E: EnsembleTrait + Sync> PTMC<E, Spherical> {
    pub fn parallel_tempering_exchange(&mut self) {
        let n_exc = rand::random_range(0..self.mc_params.trajectory_number-1);
    
        self.mc_states[n_exc].count_exc[0] += 1;
        self.mc_states[n_exc+1].count_exc[0] += 1;
    
        let energy1 = self.mc_states[n_exc].total_energy;
        let energy2 = self.mc_states[n_exc+1].total_energy;
    
        if rand::random::<NumericData>() < PTMC::<E,Spherical>::exchange_acceptance(self.mc_states[n_exc].beta, self.mc_states[n_exc+1].beta, energy1, energy2) {
            self.exchange_trajectories(n_exc);
            self.mc_states[n_exc].count_exc[1] += 1;
            self.mc_states[n_exc+1].count_exc[1] += 1;
        }
    }

    pub fn mc_step(&mut self, n_steps: usize) {
        self.mc_states.par_iter_mut().for_each(|state|
            for _ in 0..n_steps {
                state.mc_move(&self.move_strategy, &self.potential);
            }
        );
    }

    pub fn mc_cycle(&mut self, n_steps: usize, index: usize) {
        self.mc_step(n_steps);

        if rand::random::<NumericData>() < 0.1 {
            self.parallel_tempering_exchange();
        }

        if index % self.mc_params.adjust_period == 0 {
            for state in self.mc_states.iter_mut() {
                state.update_max_stepsize(self.mc_params.adjust_period, self.mc_params.min_acceptance, self.mc_params.max_acceptance, &self.ensemble);
            }
        }
    }

    pub fn update_total_energy(&mut self) {
        for mc_state in self.mc_states.iter_mut() {
            mc_state.hamiltonian.0 += mc_state.total_energy;
            mc_state.hamiltonian.1 += mc_state.total_energy*mc_state.total_energy;
        }
    }

    pub fn sampling_step(&mut self, save_index: usize, rdfsave: bool) {
        if save_index % self.mc_params.mc_sample_interval == 0 {
            self.update_total_energy();
            self.results.update_energy_histogram(&self.mc_states);

            if rdfsave {
                self.results.update_rdf(&self.mc_states);
            }
        }
    }

    pub fn equilibration_cycle(&mut self) {
        for i in 0..self.mc_params.equilibration_cycles {
            self.mc_cycle(self.n_steps, i);

            for state in self.mc_states.iter() {
                self.energy_bounds.check_energy_bounds(state.total_energy);
            }
        }

        for state in self.mc_states.iter_mut() {
            state.reset_counters();
        }
        TestContents::print(&self.energy_bounds, "energy_bounds.json");
        self.results.initialise_histograms_spherical(&self.energy_bounds, &self.mc_states[0].configuration.boundary_condition);
    }

    pub fn run(mc_params: MCParams, temp_grid: TempGrid, ensemble: E, potential: PotentialKinds, start_config: Configuration<Spherical>, rdf_save: bool) -> Self {
        let init = Initialiser::<Spherical, E>::initialise(&mc_params, &temp_grid, &ensemble, &potential, &start_config);
        let mut ptmc = PTMC {
            temp_grid,
            mc_params,
            ensemble,
            move_strategy: init.move_strategy,
            mc_states: init.mc_states,
            potential,
            results: init.results,
            energy_bounds: EnergyBounds{ min_energy: 100.0, max_energy: -100.0 },
            start_counter: init.start_counter,
            n_steps: init.n_steps,
        };

        ptmc.equilibration_cycle();

        for i in ptmc.start_counter..ptmc.mc_params.mc_cycles {
            ptmc.mc_cycle(ptmc.n_steps, i);
            if i % ptmc.mc_params.mc_sample_interval == 0 {
                ptmc.sampling_step(i, rdf_save);
            }
        }

        ptmc.results.finalise(&ptmc.mc_states, &ptmc.mc_params);
        ptmc
    }
}

impl PTMC<NPT, Periodic> {
    pub fn parallel_tempering_exchange(&mut self) {
        let n_exc = rand::random_range(1..self.mc_params.trajectory_number-1);

        self.mc_states[n_exc].count_exc[0] += 1;
        self.mc_states[n_exc+1].count_exc[0] += 1;

        let mut energy1 = self.mc_states[n_exc].total_energy;
        let mut energy2 = self.mc_states[n_exc+1].total_energy;

        energy1 += self.ensemble.pressure * self.mc_states[n_exc].configuration.boundary_condition.get_volume();
        match self.mc_states[n_exc+1].configuration.boundary_condition {
            Periodic::Cubic(Cubic {side_length}) => {
                energy2 += self.ensemble.pressure * side_length.powi(3);
            },
            Periodic::Rhombic(Rhombic {side_length, ..}) => {
                energy2 += self.ensemble.pressure * side_length.powi(3);
            }
        }

        if rand::random::<NumericData>() < PTMC::<NPT,Periodic>::exchange_acceptance(self.mc_states[n_exc].beta, self.mc_states[n_exc+1].beta, energy1, energy2) {
            self.exchange_trajectories(n_exc);
            self.mc_states[n_exc].count_exc[1] += 1;
            self.mc_states[n_exc+1].count_exc[1] += 1;
        }
    }

    pub fn mc_step(&mut self, n_steps: usize) {
        for state in self.mc_states.iter_mut() {
            for _ in 0..n_steps {
                state.mc_move(&self.move_strategy, &self.potential);
            }
        }
    }

    pub fn mc_cycle(&mut self, n_steps: usize, index: usize) {
        self.mc_step(n_steps);

        if rand::random::<NumericData>() < 0.1 {
            self.parallel_tempering_exchange();
        }

        if index % self.mc_params.adjust_period == 0 {
            for state in self.mc_states.iter_mut() {
                state.update_max_stepsize(self.mc_params.adjust_period, self.mc_params.min_acceptance, self.mc_params.max_acceptance, &self.ensemble);
            }
        }
    }

    pub fn update_total_energy(&mut self) {
        for mc_state in self.mc_states.iter_mut() {
            mc_state.hamiltonian.0 += mc_state.total_energy + self.ensemble.pressure * mc_state.configuration.boundary_condition.get_volume();
            mc_state.hamiltonian.1 += (mc_state.total_energy + self.ensemble.pressure * mc_state.configuration.boundary_condition.get_volume()) * (mc_state.total_energy + self.ensemble.pressure * mc_state.configuration.boundary_condition.get_volume());
        }
    }

    pub fn sampling_step(&mut self, save_index: usize, results: &mut Output) {
        if save_index % self.mc_params.mc_sample_interval == 0 {
            self.update_total_energy();
            results.update_volume_histogram(&self.mc_states);
        }
    }

    pub fn equilibration_cycle(&mut self, n_steps: usize) {
        self.energy_bounds = EnergyBounds{ min_energy: 100.0, max_energy: 100.0 };

        for i in 0..self.mc_params.equilibration_cycles {
            self.mc_cycle(n_steps, i);

            for state in self.mc_states.iter() {
                self.energy_bounds.check_energy_bounds(state.total_energy);
            }
        }

        for state in self.mc_states.iter_mut() {
            state.reset_counters();
        }

        match &self.mc_states[0].configuration.boundary_condition {
            Periodic::Cubic(cubic) => {
                self.results.initialise_histograms_cubic(&self.energy_bounds, &cubic);
            },
            Periodic::Rhombic(rhombic) => {
                self.results.initialise_histograms_rhombic(&self.energy_bounds, &rhombic);
            }
        }
    }
}
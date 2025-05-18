use super::{boundary_conditions::{BoundaryConditionTrait, Periodic, Spherical}, configurations::Configuration, energy_evaluation::PotentialKinds, ensembles::{EnsembleTrait, MoveStrategy}, input_params::{MCParams, Output, TempGrid}, mc_state::MCState};

pub struct Initialiser<BC: BoundaryConditionTrait, E: EnsembleTrait> {
    pub mc_states: Vec<MCState<BC>>,
    pub move_strategy: MoveStrategy<E>,
    pub results: Output,
    pub n_steps: usize,
    pub start_counter: usize,
}

impl<E: EnsembleTrait> Initialiser<Spherical, E> {
    pub fn initialise(mc_params: &MCParams, temp_grid: &TempGrid, ensemble: &E, potential: &PotentialKinds, start_config: &Configuration<Spherical>) -> Initialiser<Spherical, E> {
        let move_strategy = ensemble.get_move_strategy();
        let n_steps = move_strategy.len();
        let mc_states: Vec<MCState<Spherical>> = (0..mc_params.trajectory_number).map(|i| {
            let temperature = temp_grid.t_grid[i];
            let beta = temp_grid.beta_grid[i];
            start_config.get_mc_state(temperature, beta, ensemble, potential)
        }).collect();
        let results = Output::bin_number(mc_params);
        //results.min_energy = mc_states[0].total_energy; bug?

        Initialiser {
            mc_states,
            move_strategy,
            results,
            n_steps,
            start_counter: 1,
        }
    }
}

impl <E: EnsembleTrait> Initialiser<Periodic, E> {
    pub fn initialise(mc_params: &MCParams, temp_grid: &TempGrid, ensemble: &E, potential: &PotentialKinds, start_config: &Configuration<Periodic>) -> Initialiser<Periodic, E> {
        let move_strategy = ensemble.get_move_strategy();
        let n_steps = move_strategy.len();
        let mc_states: Vec<MCState<Periodic>> = (0..mc_params.trajectory_number).map(|i| {
            let temperature = temp_grid.t_grid[i];
            let beta = temp_grid.beta_grid[i];
            start_config.get_mc_state(temperature, beta, ensemble, potential)
        }).collect();
        let results = Output::bin_number(mc_params);
        //results.min_energy = mc_states[0].total_energy; bug?

        Initialiser {
            mc_states,
            move_strategy,
            results,
            n_steps,
            start_counter: 1,
        }
    }
}
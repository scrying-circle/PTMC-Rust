
use ndarray::{s, Array2, Array3, ArrayBase, AssignElem, Axis, Dim, OwnedRepr};
use serde::{Deserialize, Serialize};

use super::{boundary_conditions::{BoundaryConditionTrait, Cubic, Periodic, PeriodicTrait, Rhombic, Spherical}, mc_state::MCState, test_contents::TestContents, type_lib::NumericData};

pub struct MCParams {
    pub mc_cycles: usize,
    pub equilibration_cycles: usize,
    pub mc_sample_interval: usize,
    pub trajectory_number: usize,
    pub atom_number: usize,
    pub adjust_period: usize,
    pub bin_number: usize,
    pub min_acceptance: NumericData,
    pub max_acceptance: NumericData,
}

impl MCParams {
    pub fn new(mc_cycles: usize, trajectory_number: usize, atom_number: usize, mc_sample_interval: usize, adjust_period: usize) -> Self {
        MCParams {
            mc_cycles,
            equilibration_cycles: mc_cycles / 5,
            mc_sample_interval,
            trajectory_number,
            atom_number,
            adjust_period,
            bin_number: 100,
            min_acceptance: 0.4,
            max_acceptance: 0.6,
        }
    }
}

pub struct TempGrid {
    pub t_grid: Vec<NumericData>,
    pub beta_grid: Vec<NumericData>,
}

impl TempGrid {
    pub fn new_equally_spaced(t_min: NumericData, t_max: NumericData, n_traj: usize) -> Self {
        let delta = (t_max - t_min) / (n_traj - 1) as NumericData;
        let t_grid: Vec<NumericData> = (0..n_traj).map(|i| t_min + i as NumericData * delta as NumericData).collect();
        let beta_grid = t_grid.iter().map(|&t| 1.0 / t / 3.16681196e-6).collect();
        TempGrid {
            t_grid,
            beta_grid,
        }
    }

    pub fn new_geometrically_spaced(t_min: NumericData, t_max: NumericData, n_traj: usize) -> Self {
        let t_grid: Vec<NumericData> = (0..n_traj).map(|i| t_min * (t_max/t_min).powf(i as NumericData/(n_traj-1) as NumericData)).collect();
        let beta_grid = t_grid.iter().map(|&t| 1.0 / t / 3.16681196e-6).collect();
        TempGrid {
            t_grid,
            beta_grid,
        }
    }
}

pub struct Output {
    pub bin_number: usize,
    pub min_energy: NumericData,
    pub max_energy: NumericData,
    pub min_volume: NumericData,
    pub max_volume: NumericData,
    pub delta_energy_hist: NumericData,
    pub delta_volume_hist: NumericData,
    pub delta_r2: NumericData,
    pub max_displacement: Vec<NumericData>,
    pub average_energy: Vec<NumericData>,
    pub heat_capacity: Vec<NumericData>,
    pub energy_histogram: Array2<NumericData>,
    pub volume_histogram: Array3<NumericData>,
    pub rdf: Array2<NumericData>,
    pub count_stat_atom: Vec<NumericData>,
    pub count_stat_volume: Vec<NumericData>,
    pub count_stat_exc: Vec<NumericData>,
    pub count_stat_rot: Vec<NumericData>,
}

impl Output {
    pub fn bin_number(mc_params: &MCParams) -> Self {
        Output {
            bin_number: mc_params.bin_number,
            min_energy: 0.0,
            max_energy: 0.0,
            min_volume: 0.0,
            max_volume: 0.0,
            delta_energy_hist: 0.0,
            delta_volume_hist: 0.0,
            delta_r2: 0.0,
            max_displacement: Vec::new(),
            average_energy: Vec::new(),
            heat_capacity: Vec::new(),
            energy_histogram: Array2::<NumericData>::zeros((mc_params.trajectory_number, mc_params.bin_number + 2)),
            volume_histogram: Array3::<NumericData>::zeros((mc_params.trajectory_number, mc_params.bin_number + 2, mc_params.bin_number + 2)),
            rdf: Array2::<NumericData>::zeros((mc_params.trajectory_number, mc_params.bin_number * 5)),
            count_stat_atom: Vec::new(),
            count_stat_volume: Vec::new(),
            count_stat_exc: Vec::new(),
            count_stat_rot: Vec::new(),
        }
    }

    pub fn new(bin_number: usize,
        min_energy: NumericData,
        max_energy: NumericData,
        min_volume: NumericData,
        max_volume: NumericData,
        max_displacement: Vec<NumericData>,
        average_energy: Vec<NumericData>,
        heat_capacity: Vec<NumericData>,
        energy_histogram: Array2<NumericData>,
        volume_histogram: Array3<NumericData>,
        rdf: Array2<NumericData>,
        count_stat_atom: Vec<NumericData>,
        count_stat_volume: Vec<NumericData>,
        count_stat_exc: Vec<NumericData>,
        count_stat_rot: Vec<NumericData>,
    ) -> Self {
        Output {
            bin_number,
            min_energy,
            max_energy,
            min_volume,
            max_volume,
            delta_energy_hist: (max_energy - min_energy) / (bin_number-1) as NumericData,
            delta_volume_hist: (max_volume - min_volume) / bin_number as NumericData,
            delta_r2: 0.0,
            max_displacement,
            average_energy,
            heat_capacity,
            energy_histogram,
            volume_histogram,
            rdf,
            count_stat_atom,
            count_stat_volume,
            count_stat_exc,
            count_stat_rot,
        }
    }

    pub fn initialise_histograms_spherical(&mut self, energy_bounds: &EnergyBounds, bc: &Spherical) {
        self.min_energy = energy_bounds.min_energy;
        self.max_energy = energy_bounds.max_energy;

        self.delta_energy_hist = (self.max_energy - self.min_energy) / (self.bin_number-1) as NumericData;
        self.delta_r2 = bc.radius2 / 0.8 / self.bin_number as NumericData;
    }

    pub fn initialise_histograms_cubic(&mut self, energy_bounds: &EnergyBounds, bc: &Cubic) {
        self.min_energy = energy_bounds.min_energy;
        self.max_energy = energy_bounds.max_energy;

        self.min_volume = bc.side_length.powi(3)*0.8;
        self.max_volume = bc.side_length.powi(3)*2.0;

        self.delta_energy_hist = (self.max_energy - self.min_energy) / (self.bin_number-1) as NumericData;
        self.delta_volume_hist = (self.max_volume - self.min_volume) / self.bin_number as NumericData;
    }

    pub fn initialise_histograms_rhombic(&mut self, energy_bounds: &EnergyBounds, bc: &Rhombic) {
        self.min_energy = energy_bounds.min_energy;
        self.max_energy = energy_bounds.max_energy;

        self.min_volume = bc.side_length*bc.side_length * bc.side_height * 3_f32.sqrt()/2.0*0.8;
        self.max_volume = self.min_volume * 2.0 / 0.8;

        self.delta_energy_hist = (self.max_energy - self.min_energy) / (self.bin_number-1) as NumericData;
        self.delta_volume_hist = (self.max_volume - self.min_volume) / self.bin_number as NumericData;
    }

    pub fn get_histogram_index(histogram_number: NumericData, bin_number: usize) -> usize {
        if histogram_number < 1.0 {
            1
        } else if histogram_number > bin_number as NumericData {
            bin_number + 2
        } else {
            histogram_number.floor() as usize + 1
        }
    }

    pub fn find_energy_histogram_index<T: BoundaryConditionTrait>(&self, mc_state: &MCState<T>) -> usize {
        Output::get_histogram_index((mc_state.total_energy - self.min_energy)/self.delta_energy_hist + 1.0, self.bin_number) - 1
    }

    pub fn find_volume_histogram_index(&self, mc_state: &MCState<Periodic>) -> usize {
        Output::get_histogram_index((mc_state.configuration.boundary_condition.get_side_length().powi(3) - self.min_volume)/self.delta_volume_hist + 1.0, self.bin_number)
    }
    
    pub fn update_volume_histogram(&mut self, mc_states: &Vec<MCState<Periodic>>) {
        for i in 0..mc_states.len() {
            let en_hist_index = self.find_energy_histogram_index(&mc_states[i]);
            let vol_hist_index = self.find_volume_histogram_index(&mc_states[i]);
            self.volume_histogram[ [i, en_hist_index, vol_hist_index] ] += 1.0;
        }
    }

    pub fn update_energy_histogram<T: BoundaryConditionTrait>(&mut self, mc_states: &Vec<MCState<T>>) {
        for i in 0..mc_states.len() {
            let en_hist_index = self.find_energy_histogram_index(&mc_states[i]);
            self.energy_histogram[ [i, en_hist_index] ] += 1.0;
        }
    }

    pub fn update_rdf<T: BoundaryConditionTrait>(&mut self, mc_states: &Vec<MCState<T>>) {
        for trajectory in 0..mc_states.len() {
            for i in 0..mc_states[1].configuration.position_vector.len() {
                for k in 0..i {
                    let index = (mc_states[trajectory].distance2_mat[ [i, k] ]/self.delta_r2).floor() as usize;
                    if index != 0 && index <= self.bin_number * 5 {
                        self.rdf[ [trajectory, index] ] += 1.0;
                    }
                }
            }
        }
    }

    pub fn finalise<T: BoundaryConditionTrait>(&mut self, mc_states: &Vec<MCState<T>>, mc_params: &MCParams) {
        let n_sample = mc_params.mc_cycles / mc_params.mc_sample_interval;
        self.average_energy = mc_states.iter().map(|state| state.hamiltonian.0/n_sample as NumericData).collect();
        let energy_squared_average: Vec<NumericData> = mc_states.iter().map(|state| state.hamiltonian.1/n_sample as NumericData).collect();

        self.heat_capacity = mc_states.iter().zip(self.average_energy.iter()).zip(energy_squared_average.iter()).map(|((state, &energy), &energy_squared)| (energy_squared - energy*energy) * state.beta * state.beta).collect();

        self.count_stat_atom = mc_states.iter().map(|state| state.count_atom[0] as NumericData / (mc_params.atom_number * mc_params.mc_cycles) as NumericData).collect();

        self.count_stat_exc = mc_states.iter().map(|state| state.count_exc[1] as NumericData / state.count_exc[0] as NumericData).collect();
    }

    pub fn readfile(&self) -> Vec<NumericData> {
        let de = (self.max_energy - self.min_energy) / (self.bin_number -1) as NumericData;
        (0..self.bin_number).map(|i| self.min_energy + i as NumericData * de).collect::<Vec<_>>()
    }

    pub fn process_histogram(&mut self, energy_vector: &Vec<NumericData>) -> (Array2<NumericData>, Vec<NumericData>, Vec<NumericData>) {
        let mut sums = self.energy_histogram.axis_iter(Axis(1))
        .map(|row| row.sum()/self.bin_number as NumericData)
        .collect::<Vec<_>>();

        let mut mask = sums.iter()
        .map(|value| *value != 0.0)
        .collect::<Vec<bool>>();

        mask.first_mut().unwrap().assign_elem(false);
        mask.last_mut().unwrap().assign_elem(false);

        let mut processed_histogram = self.energy_histogram.select(Axis(1), &mask.iter().enumerate()
        .filter_map(|(i, &keep)| keep.then_some(i))
        .collect::<Vec<_>>());

        for item in processed_histogram.iter_mut() {
            *item /= self.bin_number as NumericData;
        }

        let processed_energy_vector = mask.iter().skip(1).take(self.bin_number).enumerate()
        .filter_map(|(i, &keep)| keep.then_some(energy_vector[i])).collect::<Vec<_>>();
        TestContents::print(&processed_energy_vector, "processed_energy_vector.json");
        let mut iter = mask.iter();
        sums.retain(|_| *iter.next().unwrap());
        println!("{:?}", processed_histogram);
        (processed_histogram, processed_energy_vector, sums)
        
    }
}

#[derive(Debug)]
#[derive(Serialize, Deserialize)]
pub struct EnergyBounds {
    pub min_energy: NumericData,
    pub max_energy: NumericData,
}

impl EnergyBounds {
    pub fn check_energy_bounds(&mut self, energy: NumericData) {
        if energy < self.min_energy {
            self.min_energy = energy;
        } else if energy > self.max_energy {
            self.max_energy = energy;
        }
    }
}
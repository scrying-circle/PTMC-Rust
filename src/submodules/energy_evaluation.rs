use std::{f32::consts::PI, ops::Index};

use enum_dispatch::enum_dispatch;
use ndarray::{s, Array1, Array2};

use super::{boundary_conditions::{BoundaryConditionTrait, PeriodicTrait}, configurations::Configuration, type_lib::{NumericData, Position}};

pub struct ELJEven {
    pub coeffs: Vec<NumericData>,
}

pub enum PotentialKinds {
    ELJ(ELJ),
    ELJEven(ELJEven),
    ELJB(ELJB),
}

impl Dimer for ELJEven {
    fn dimer_energy(&self, r2: NumericData) -> NumericData {
        let mut r6inv = 1.0/(r2*r2*r2);
        let mut sum = 0.0;
        for coeff in self.coeffs.iter() {
            sum += coeff * r6inv;
            r6inv /= r2;
        }
        sum
    }

    fn long_range_correction(&self, n_atoms: usize, r_cut: NumericData) -> NumericData {
        let r_cut_sqrt = r_cut.sqrt();
        let mut rc3 = r_cut*r_cut_sqrt;
        let mut e_lrc = 0.0;
        for i in 0..self.coeffs.len() {
            e_lrc += self.coeffs[i] / rc3 / (2*i+1) as NumericData;
            rc3 *= r_cut;
        }
        e_lrc * PI * (n_atoms * n_atoms) as NumericData / 4.0 / r_cut_sqrt*r_cut_sqrt*r_cut_sqrt
    }
}

pub struct ELJ {
    pub coeffs: Vec<NumericData>,
}

impl Dimer for ELJ {
    fn dimer_energy(&self, r2: NumericData) -> NumericData {
        let r = r2.sqrt();
        let mut r6inv = 1.0/(r2*r2*r2);
        let mut sum = 0.0;
        for coeff in self.coeffs.iter() {
            sum += coeff * r6inv;
            r6inv /= r;
        }
        sum
    }

    fn long_range_correction(&self, n_atoms: usize, r_cut: NumericData) -> NumericData {
        let r_cut_sqrt = r_cut.sqrt();
        let mut rc3 = r_cut*r_cut_sqrt;
        let mut e_lrc = 0.0;
        for i in 0..self.coeffs.len() {
            e_lrc += self.coeffs[i] / rc3 / (2*i+1) as NumericData;
            rc3 *= r_cut;
        }
        e_lrc * PI * (n_atoms * n_atoms) as NumericData / 4.0 / r_cut_sqrt*r_cut_sqrt*r_cut_sqrt
    }
}

pub trait Dimer {
    fn dimer_energy(&self, r2: NumericData) -> NumericData;

    fn dimer_energy_atom<T: Index<usize, Output = NumericData>>(&self, atom_index: usize, d2vec: &T, n_atoms: usize) -> NumericData {
        let mut sum = 0.0;
        for i in (0..n_atoms).filter(|&i| i != atom_index) {
            sum += self.dimer_energy(d2vec[i]);
        }
        sum
    }

    fn dimer_energy_atom_r_cut<T: Index<usize, Output = NumericData>>(&self, atom_index: usize, d2vec: &T, n_atoms: usize, r_cut: NumericData) -> NumericData {
        let mut sum = 0.0;
        for i in (0..n_atoms).filter(|&i| i != atom_index || d2vec[i] <= r_cut) {
            sum += self.dimer_energy(d2vec[i]);
        }
        sum
    }

    fn dimer_energy_config_periodic<T: PeriodicTrait>(&self, distance_matrix: &Array2<NumericData>, bc: &T, r_cut: NumericData) -> (Vec<NumericData>, NumericData) {
        let n_atoms = distance_matrix.shape()[0];
        let mut dimer_energy_vec = vec![0.0; n_atoms];
        let mut energy_total = 0.0;

        for i in 0..n_atoms {
            for j in i+1..n_atoms {
                let r2 = distance_matrix[[i, j]];
                if r2 <= r_cut {
                    let e_ij = self.dimer_energy(r2);
                    dimer_energy_vec[i] += e_ij;
                    dimer_energy_vec[j] += e_ij;
                    energy_total += e_ij;
                }
            }
        }
        (dimer_energy_vec, energy_total + self.long_range_correction(n_atoms, r_cut) * bc.get_lrc_scale_factor())
    }

    fn dimer_energy_config_aperiodic(&self, distance_matrix: &Array2<NumericData>) -> (Vec<NumericData>, NumericData) {
        let n_atoms = distance_matrix.shape()[0];
        let mut dimer_energy_vec = vec![0.0; n_atoms];
        let mut energy_total = 0.0;

        for i in 0..n_atoms {
            for j in i+1..n_atoms {
                let e_ij = self.dimer_energy(distance_matrix[[i, j]]);
                dimer_energy_vec[i] += e_ij;
                dimer_energy_vec[j] += e_ij;
                energy_total += e_ij;
            }
        }
        (dimer_energy_vec, energy_total)
    }

    fn dimer_energy_update(&self, atom_index: usize, distance_matrix: &Array2<NumericData>, new_dist_vec: &Array1<NumericData>) -> NumericData {
        let n_atoms = new_dist_vec.len();
        self.dimer_energy_atom(atom_index, new_dist_vec, n_atoms) - self.dimer_energy_atom(atom_index, &distance_matrix.slice(s![atom_index, ..]), n_atoms)
    }

    fn dimer_energy_update_r_cut(&self, atom_index: usize, distance_matrix: &Array2<NumericData>, new_dist_vec: &Array1<NumericData>, r_cut: NumericData) -> NumericData {
        let n_atoms = new_dist_vec.len();
        self.dimer_energy_atom_r_cut(atom_index, new_dist_vec, n_atoms, r_cut) - self.dimer_energy_atom_r_cut(atom_index, &distance_matrix.slice(s![atom_index, ..]), n_atoms, r_cut)
    }

    fn long_range_correction(&self, n_atoms: usize, r_cut: NumericData) -> NumericData;

    fn energy_update(&self, atom_index: usize, distance_matrix: &Array2<NumericData>, new_dist_vec: &Array1<NumericData>) -> NumericData {
        self.dimer_energy_update(atom_index, distance_matrix, new_dist_vec)
    }

    fn energy_update_r_cut(&self, atom_index: usize, distance_matrix: &Array2<NumericData>, new_dist_vec: &Array1<NumericData>, r_cut: NumericData) -> NumericData {
        self.dimer_energy_update_r_cut(atom_index, distance_matrix, new_dist_vec, r_cut)
    }

    fn initialise_energy_periodic<T: PeriodicTrait>(&self, config: &Configuration<T>, distance_matrix: &Array2<NumericData>, potential_variables: &mut DimerPotentialVariables, r_cut: NumericData) -> NumericData{
        let dimer_config = self.dimer_energy_config_periodic(distance_matrix, &config.boundary_condition, r_cut);
        potential_variables.set_en_atom_vec(dimer_config.0);
        dimer_config.1
    }

    fn initialise_energy_aperiodic(&self, distance_matrix: &Array2<NumericData>, potential_variables: &mut DimerPotentialVariables) -> NumericData{
        let dimer_config = self.dimer_energy_config_aperiodic(distance_matrix);
        potential_variables.set_en_atom_vec(dimer_config.0);
        dimer_config.1
    }

    fn set_variables(&self, n_atoms: usize) -> DimerPotentialVariables {
        DimerPotentialVariables { en_atom_vec: vec![0.0; n_atoms] }
    }
}

pub struct ELJB {
    pub a: Vec<NumericData>,
    pub b: Vec<NumericData>,
    pub c: Vec<NumericData>,
}

impl MagneticDimer for ELJB {
    fn dimer_energy(&self, r2: NumericData, z_angle: NumericData) -> NumericData {
        if r2 >= 5.3 {
            let mut r6inv = 1.0/(r2*r2*r2);
            let t2 = 2.0/(z_angle*z_angle+1.0)-1.0;
            let t4 = 2.0*t2*t2-1.0;
            let mut sum = self.c[0] * r6inv * (1.0 + self.a[0] * t2 + self.b[0] * t4);
            r6inv /= r2;
            for i in 1..self.a.len() {
                sum += self.c[i] * r6inv * (1.0 + self.a[0] * t2 + self.b[0] * t4);
                r6inv /= r2.sqrt();
            }
            sum
        } else {
            0.1
        }
    }

    fn long_range_correction(&self, n_atoms: usize, r_cut: NumericData) -> NumericData {
        let coeffs = vec![-0.1279111890228638, -1.328138539967966, 12.260941135261255,41.1221240825166];
        let r_cut_sqrt = r_cut.sqrt();
        let mut rc3 = r_cut*r_cut_sqrt;
        let mut e_lrc = 0.0;
        for i in 0..4 {
            e_lrc += coeffs[i] / rc3 / (2*i+1) as NumericData;
            rc3 *= r_cut;
        }
        e_lrc * PI * (n_atoms * n_atoms) as NumericData / 4.0 / r_cut_sqrt*r_cut_sqrt*r_cut_sqrt
    }
}

pub trait MagneticDimer {
    fn dimer_energy(&self, r2: NumericData, z_angle: NumericData) -> NumericData;

    fn dimer_energy_atom<T: Index<usize, Output = NumericData>>(&self, atom_index: usize, d2vec: &T, tan_vec: &T, n_atoms: usize) -> NumericData {
        let mut sum = 0.0;
        for i in (0..n_atoms).filter(|&i| i != atom_index) {
            sum += self.dimer_energy(d2vec[i], tan_vec[i]);
        }
        sum
    }

    fn dimer_energy_atom_r_cut<T: Index<usize, Output = NumericData>>(&self, atom_index: usize, d2vec: &T, tan_vec: &T, n_atoms: usize, r_cut: NumericData) -> NumericData {
        let mut sum = 0.0;
        for i in (0..n_atoms).filter(|&i| i != atom_index || d2vec[i] <= r_cut) {
            sum += self.dimer_energy(d2vec[i], tan_vec[i]);
        }
        sum
    }

    fn dimer_energy_config_periodic<T: PeriodicTrait>(&self, distance_matrix: &Array2<NumericData>, bc: &T, r_cut: NumericData, potential_variables: &MagneticELJVariables) -> (Vec<NumericData>, NumericData) {
        let n_atoms = distance_matrix.shape()[0];
        let mut dimer_energy_vec = vec![0.0; n_atoms];
        let mut energy_total = 0.0;

        for i in 0..n_atoms {
            for j in i+1..n_atoms {
                let r2 = distance_matrix[[i, j]];
                if r2 <= r_cut {
                    let e_ij = self.dimer_energy(r2, potential_variables.tan_mat[[i, j]]);
                    dimer_energy_vec[i] += e_ij;
                    dimer_energy_vec[j] += e_ij;
                    energy_total += e_ij;
                }
            }
        }
        (dimer_energy_vec, energy_total + self.long_range_correction(n_atoms, r_cut) * bc.get_lrc_scale_factor())
    }

    fn dimer_energy_config_aperiodic(&self, distance_matrix: &Array2<NumericData>, potential_variables: &MagneticELJVariables) -> (Vec<NumericData>, NumericData) {
        let n_atoms = distance_matrix.shape()[0];
        let mut dimer_energy_vec = vec![0.0; n_atoms];
        let mut energy_total = 0.0;

        for i in 0..n_atoms {
            for j in i+1..n_atoms {
                let e_ij = self.dimer_energy(distance_matrix[[i, j]], potential_variables.tan_mat[[i, j]]);
                dimer_energy_vec[i] += e_ij;
                dimer_energy_vec[j] += e_ij;
                energy_total += e_ij;
            }
        }
        (dimer_energy_vec, energy_total)
    }

    fn dimer_energy_update(&self, atom_index: usize, distance_matrix: &Array2<NumericData>, new_dist_vec: &Array1<NumericData>, tan_matrix: &Array2<NumericData>, new_tan_vec: &Array1<NumericData>) -> NumericData {
        let n_atoms = new_dist_vec.len();
        self.dimer_energy_atom(atom_index, new_dist_vec, new_tan_vec, n_atoms) - self.dimer_energy_atom(atom_index, &distance_matrix.slice(s![atom_index, ..]), &tan_matrix.slice(s![atom_index, ..]), n_atoms)
    }

    fn dimer_energy_update_r_cut(&self, atom_index: usize, distance_matrix: &Array2<NumericData>, new_dist_vec: &Array1<NumericData>, tan_matrix: &Array2<NumericData>, new_tan_vec: &Array1<NumericData>, r_cut: NumericData) -> NumericData {
        let n_atoms = new_dist_vec.len();
        self.dimer_energy_atom_r_cut(atom_index, new_dist_vec, new_tan_vec, n_atoms, r_cut) - self.dimer_energy_atom_r_cut(atom_index, &distance_matrix.slice(s![atom_index, ..]), &tan_matrix.slice(s![atom_index, ..]), n_atoms, r_cut)
    }

    fn long_range_correction(&self, n_atoms: usize, r_cut: NumericData) -> NumericData;

    fn energy_update<T: BoundaryConditionTrait>(&self, atom_index: usize, distance_matrix: &Array2<NumericData>, new_dist_vec: &Array1<NumericData>, potential_variables: &mut MagneticELJVariables, trial_move: &Position, config: &Configuration<T>) -> NumericData {
        potential_variables.new_tan_vec = config.position_vector.iter().map(|pos| config.boundary_condition.get_tan(trial_move, pos)).collect();
        potential_variables.new_tan_vec[atom_index] = 0.0;
        self.dimer_energy_update(atom_index, distance_matrix, new_dist_vec, &potential_variables.tan_mat, &potential_variables.new_tan_vec)
    }

    fn energy_update_r_cut<T: BoundaryConditionTrait>(&self, atom_index: usize, distance_matrix: &Array2<NumericData>, new_dist_vec: &Array1<NumericData>, potential_variables: &mut MagneticELJVariables, trial_move: &Position, config: &Configuration<T>, r_cut: NumericData) -> NumericData {
        potential_variables.new_tan_vec = config.position_vector.iter().map(|pos| config.boundary_condition.get_tan(trial_move, pos)).collect();
        potential_variables.new_tan_vec[atom_index] = 0.0;
        self.dimer_energy_update_r_cut(atom_index, distance_matrix, new_dist_vec, &potential_variables.tan_mat, &potential_variables.new_tan_vec, r_cut)
    }

    fn initialise_energy_aperiodic(&self, distance_matrix: &Array2<NumericData>, potential_variables: &mut MagneticELJVariables) -> NumericData {
        let dimer_config = self.dimer_energy_config_aperiodic(distance_matrix, potential_variables);
        potential_variables.set_en_atom_vec(dimer_config.0);
        dimer_config.1
    }

    fn initialise_energy_periodic<T: PeriodicTrait>(&self, config: &Configuration<T>, distance_matrix: &Array2<NumericData>, potential_variables: &mut MagneticELJVariables, r_cut: NumericData) -> NumericData {
        let dimer_config = self.dimer_energy_config_periodic(distance_matrix, &config.boundary_condition, r_cut, potential_variables);
        potential_variables.set_en_atom_vec(dimer_config.0);
        dimer_config.1
    }

    fn set_variables<BC: BoundaryConditionTrait>(&self, config: &Configuration<BC>) -> MagneticELJVariables {
        let tan_mat = config.get_tan_mat();
        let n_atoms = config.number_of_atoms;
        MagneticELJVariables { en_atom_vec: vec![0.0; n_atoms], new_tan_vec: Array1::<NumericData>::zeros(n_atoms), tan_mat }
    }
}

#[enum_dispatch]
pub trait PotentialVariablesTrait {
    fn get_en_atom_vec(&self) -> &Vec<NumericData>;
    fn set_en_atom_vec(&mut self, new_en_vec: Vec<NumericData>);
    fn swap_vars(&mut self, atom_index: usize);
}
pub struct MagneticELJVariables {
    en_atom_vec: Vec<NumericData>,
    new_tan_vec: Array1<NumericData>,
    tan_mat: Array2<NumericData>,
}

impl PotentialVariablesTrait for MagneticELJVariables {
    fn get_en_atom_vec(&self) -> &Vec<NumericData> {
        &self.en_atom_vec
    }

    fn set_en_atom_vec(&mut self, new_en_vec: Vec<NumericData>) {
        self.en_atom_vec = new_en_vec;
    }

    fn swap_vars(&mut self, atom_index: usize) {
        self.tan_mat.row_mut(atom_index).assign(&self.new_tan_vec);
        self.tan_mat.column_mut(atom_index).assign(&self.new_tan_vec);
    }
}

#[enum_dispatch(PotentialVariablesTrait)]
pub enum PotentialVariableKinds {
    Dimer(DimerPotentialVariables),
    Magnetic(MagneticELJVariables),
}

pub struct DimerPotentialVariables {
    pub en_atom_vec: Vec<NumericData>,
}

impl PotentialVariablesTrait for DimerPotentialVariables {
    fn get_en_atom_vec(&self) -> &Vec<NumericData> {
        &self.en_atom_vec
    }

    fn set_en_atom_vec(&mut self, new_en_vec: Vec<NumericData>) {
        self.en_atom_vec = new_en_vec;
    }

    fn swap_vars(&mut self, _atom_index: usize) {
        ()
    }
}
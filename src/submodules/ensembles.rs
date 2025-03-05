use enum_dispatch::enum_dispatch;
use ndarray::Array2;

use super::{boundary_conditions::{Periodic, PeriodicTrait}, configurations::Configuration, type_lib::{NumericData, Position}};

#[derive(Clone)]
pub struct NVT {
    pub n_atoms: usize,
    pub n_atom_moves: usize,
    pub n_atom_swaps: usize,
}

impl EnsembleTrait for NVT {
    fn set_ensemble_variables(&self, _config: Option<&Configuration<Periodic>>) -> EnsembleVariableKinds {
        EnsembleVariableKinds::NVT(NVTVariables{ atom_index: 1, trial_move: [0.0; 3] })
    }

    fn get_move_strategy(&self) -> MoveStrategy<Self> {
        let mut move_type_vec = Vec::with_capacity((self.n_atom_moves + self.n_atom_swaps) as usize);
        for _ in 0..self.n_atom_moves {
            move_type_vec.push(MoveType::AtomMove);
        }
        for _ in 0..self.n_atom_swaps {
            move_type_vec.push(MoveType::AtomSwap);
        }
        MoveStrategy { ensemble: self.clone(), move_type_vec }
    }

    fn get_n_atom_moves(&self) -> usize {
        self.n_atom_moves
    }
}

impl NVT {
    pub fn new(n_atoms: usize) -> Self {
        NVT { n_atoms, n_atom_moves: n_atoms, n_atom_swaps: 0 }
    }
}

#[derive(Clone)]
pub struct NPT {
    pub n_atoms: usize,
    pub n_atom_moves: usize,
    pub n_volume_moves: usize,
    pub n_atom_swaps: usize,
    pub pressure: NumericData,
}

impl EnsembleTrait for NPT {
    fn set_ensemble_variables(&self, option_config: Option<&Configuration<Periodic>>) -> EnsembleVariableKinds {
        let trial_move = [0.0; 3];
        let config = option_config.unwrap();
        let trial_config = config.clone();
        let new_dist2_mat = Array2::<NumericData>::zeros((trial_config.number_of_atoms, trial_config.number_of_atoms));
        let r_cut = config.boundary_condition.get_r_cut();
        let new_r_cut = 0.0;
        EnsembleVariableKinds::NPT(NPTVariables { atom_index: 1, trial_move, trial_config, new_dist2_mat, r_cut, new_r_cut })
    }

    fn get_move_strategy(&self) -> MoveStrategy<Self> {
        let mut move_type_vec = Vec::with_capacity((self.n_atom_moves + self.n_atom_swaps) as usize);
        for _ in 0..self.n_atom_moves {
            move_type_vec.push(MoveType::AtomMove);
        }
        for _ in 0..self.n_volume_moves {
            move_type_vec.push(MoveType::VolumeMove);
        }
        for _ in 0..self.n_atom_swaps {
            move_type_vec.push(MoveType::AtomSwap);
        }

        MoveStrategy { ensemble: self.clone(), move_type_vec }
    }

    fn get_n_atom_moves(&self) -> usize {
        self.n_atom_moves
    }
}

impl NPT {
    pub fn new(n_atoms: usize, pressure: NumericData) -> Self {
        NPT { n_atoms, n_atom_moves: n_atoms, n_volume_moves: 1, n_atom_swaps: 0, pressure }
    }
}

#[derive(Clone)]
pub struct NNVT {
    pub n_atoms: (usize, usize),
    pub n_atom_moves: usize,
    pub n_atom_swaps: usize,
}

impl EnsembleTrait for NNVT {
    fn set_ensemble_variables(&self, _config: Option<&Configuration<Periodic>>) -> EnsembleVariableKinds {
        EnsembleVariableKinds::NNVT(NNVTVariables{ atom_index: 1, trial_move: [0.0; 3], swap_indices: (1, self.n_atoms.0+1), n1n2: (self.n_atoms.0, self.n_atoms.1) }) 
    }

    fn get_move_strategy(&self) -> MoveStrategy<Self> {
        let mut move_type_vec = Vec::with_capacity((self.n_atom_moves + self.n_atom_swaps) as usize);
        for _ in 0..self.n_atom_moves {
            move_type_vec.push(MoveType::AtomMove);
        }
        for _ in 0..self.n_atom_swaps {
            move_type_vec.push(MoveType::AtomSwap);
        }
        MoveStrategy { ensemble: self.clone(), move_type_vec }
    }

    fn get_n_atom_moves(&self) -> usize {
        self.n_atom_moves
    }
}

impl NNVT {
    pub fn new(n_atoms: (usize, usize)) -> Self {
        NNVT { n_atoms, n_atom_moves: n_atoms.0 + n_atoms.1, n_atom_swaps: 1 }
    }
}

#[enum_dispatch(EnsembleVariablesTrait)]
pub enum EnsembleVariableKinds {
    NVT(NVTVariables),
    NPT(NPTVariables),
    NNVT(NNVTVariables),
}

pub trait EnsembleTrait {
    fn set_ensemble_variables(&self, config: Option<&Configuration<Periodic>>) -> EnsembleVariableKinds;
    fn get_move_strategy(&self) -> MoveStrategy<Self> where Self: Sized;
    fn get_n_atom_moves(&self) -> usize;
}

pub struct NVTVariables {
    pub atom_index: usize,
    pub trial_move: Position
}
impl EnsembleVariablesTrait for NVTVariables {
    fn trial_move(&self) -> &Position {
        &self.trial_move
    }

    fn atom_index(&self) -> usize {
        self.atom_index
    }

    fn set_trial_move(&mut self, trial_move: Position) {
        self.trial_move = trial_move;
    }

    fn set_atom_index(&mut self, atom_index: usize) {
        self.atom_index = atom_index;
    }
}

pub struct NPTVariables {
    pub atom_index: usize,
    pub trial_move: Position,
    pub trial_config: Configuration<Periodic>,
    pub new_dist2_mat: Array2<NumericData>,
    pub r_cut: NumericData,
    pub new_r_cut: NumericData
}
impl EnsembleVariablesTrait for NPTVariables {
    fn trial_move(&self) -> &Position {
        &self.trial_move
    }

    fn atom_index(&self) -> usize {
        self.atom_index
    }

    fn set_trial_move(&mut self, trial_move: Position) {
        self.trial_move = trial_move;
    }

    fn set_atom_index(&mut self, atom_index: usize) {
        self.atom_index = atom_index;
    }
}

pub struct NNVTVariables {
    pub atom_index: usize,
    pub trial_move: Position,
    pub swap_indices: (usize, usize),
    pub n1n2: (usize, usize),
}
impl EnsembleVariablesTrait for NNVTVariables {
    fn trial_move(&self) -> &Position {
        &self.trial_move
    }

    fn atom_index(&self) -> usize {
        self.atom_index
    }

    fn set_trial_move(&mut self, trial_move: Position) {
        self.trial_move = trial_move;
    }

    fn set_atom_index(&mut self, atom_index: usize) {
        self.atom_index = atom_index;
    }
}

#[enum_dispatch]
pub trait EnsembleVariablesTrait {
    fn trial_move(&self) -> &Position;
    fn atom_index(&self) -> usize;
    fn set_trial_move(&mut self, trial_move: Position);
    fn set_atom_index(&mut self, atom_index: usize);
}

pub enum MoveType {
    AtomMove,
    VolumeMove,
    AtomSwap
}

pub struct MoveStrategy<E: EnsembleTrait> {
    pub ensemble: E,
    pub move_type_vec: Vec<MoveType>,
}

impl<E: EnsembleTrait> MoveStrategy<E> {

    pub fn len(&self) -> usize {
        self.move_type_vec.len()
    }
}
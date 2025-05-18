use enum_dispatch::enum_dispatch;
use rand::random;

use super::func_lib::get_tan;
use super::type_lib::NumericData;
use super::type_lib::Position;
use super::func_lib::distance_squared;

#[enum_dispatch]
pub trait BoundaryConditionTrait: Clone {
    fn distance_squared(&self, pos1: &Position, pos2: &Position) -> NumericData;
    fn get_tan(&self, pos1: &Position, pos2: &Position) -> NumericData;
    fn atom_displacement(&self, pos: &Position, max_displacement: NumericData) -> Position;
    fn max_length(&self) -> NumericData;
}

#[derive(Clone)]
pub struct Spherical {
    pub radius2: NumericData,
}

pub trait AperiodicTrait: BoundaryConditionTrait {
    fn check_boundary(&self, pos: &Position) -> bool;
}

impl AperiodicTrait for Spherical {
    fn check_boundary(&self,pos: &Position) -> bool {
        pos[0]*pos[0] + pos[1]*pos[1] + pos[2]*pos[2] > self.radius2
    }
}

impl BoundaryConditionTrait for Spherical {
    fn distance_squared(&self, pos1: &Position, pos2: &Position) -> NumericData {
        distance_squared(pos1, pos2)
    }

    fn get_tan(&self, pos1: &Position, pos2: &Position) -> NumericData {
        get_tan(pos1, pos2)
    }

    fn max_length(&self) -> NumericData {
        30.0
    }

    fn atom_displacement(&self,pos: &Position,max_displacement:NumericData) -> Position {
        [
            pos[0] + max_displacement*(random::<NumericData>()-0.5),
            pos[1] + max_displacement*(random::<NumericData>()-0.5),
            pos[2] + max_displacement*(random::<NumericData>()-0.5),
        ]
    }
}

#[derive(Clone)]
pub struct Cubic {
    pub side_length: NumericData,
}
impl BoundaryConditionTrait for Cubic {
    fn distance_squared(&self,pos1: &Position,pos2: &Position) -> NumericData {
        let adjusted_pos2_vec = (0..3).map(|i| pos2[i] + self.side_length * ((pos1[i] - pos2[i])/self.side_length).round() ).collect::<Vec<_>>();
        let adjusted_pos2: [f32; 3] = adjusted_pos2_vec.try_into().expect("Expected a Vec of length 3");
        distance_squared(pos1, &adjusted_pos2)
    }

    fn get_tan(&self,pos1: &Position,pos2: &Position) -> NumericData {
        let adjusted_pos2_vec = (0..3).map(|i| pos2[i] + self.side_length * ((pos1[i] - pos2[i])/self.side_length).round() ).collect::<Vec<_>>();
        let adjusted_pos2: [f32; 3] = adjusted_pos2_vec.try_into().expect("Expected a Vec of length 3");
        get_tan(pos1, &adjusted_pos2)
    }

    fn max_length(&self) -> NumericData {
        self.side_length/1.8
    }

    fn atom_displacement(&self,pos: &Position,max_displacement:NumericData) -> Position {
        let trial_pos = [
            pos[0] + max_displacement*(random::<NumericData>()-0.5),
            pos[1] + max_displacement*(random::<NumericData>()-0.5),
            pos[2] + max_displacement*(random::<NumericData>()-0.5),
        ];
        [
            trial_pos[0] - self.side_length * (trial_pos[0]/self.side_length+0.5).floor(),
            trial_pos[1] - self.side_length * (trial_pos[1]/self.side_length+0.5).floor(),
            trial_pos[2] - self.side_length * (trial_pos[2]/self.side_length+0.5).floor(),
        ]
    }
}

impl PeriodicTrait for Cubic {
    fn get_volume(&self) -> NumericData {
        self.side_length.powi(3)
    }

    fn get_r_cut(&self) -> NumericData {
        self.side_length*self.side_length/4.0
    }

    fn get_side_length(&self) -> NumericData{
        self.side_length
    }
}

#[derive(Clone)]
pub struct Rhombic {
    pub side_length: NumericData,
    pub side_height: NumericData,
}
impl BoundaryConditionTrait for Rhombic {
    fn distance_squared(&self,pos1: &Position,pos2: &Position) -> NumericData {
        let adjusted_pos2: Position = [
            pos2[1] + (3_f32.powf(0.5)/2.0*self.side_length) * ((pos1[1] - pos2[1])/(3_f32.powf(0.5)/2.0*self.side_length)).round(),
            pos2[0] - pos2[1]/3_f32.powf(0.5) + self.side_length*(((pos1[0] - pos2[0])-1.0/3_f32.powf(0.5)*(pos1[1] - pos2[1]))/self.side_length).round(),
            pos2[2] + self.side_height * ((pos1[2] - pos2[2])/self.side_height).round(),
        ];
        distance_squared(pos1, &adjusted_pos2)
    }

    fn get_tan(&self,pos1: &Position,pos2: &Position) -> NumericData {
        let adjusted_pos2: Position = [
            pos2[1] + (3_f32.powf(0.5)/2.0*self.side_length) * ((pos1[1] - pos2[1])/(3_f32.powf(0.5)/2.0*self.side_length)).round(),
            pos2[0] - pos2[1]/3_f32.powf(0.5) + self.side_length*(((pos1[0] - pos2[0])-1.0/3_f32.powf(0.5)*(pos1[1] - pos2[1]))/self.side_length).round(),
            pos2[2] + self.side_height * ((pos1[2] - pos2[2])/self.side_height).round(),
        ];
        get_tan(pos1, &adjusted_pos2)
    }

    fn max_length(&self) -> NumericData {
        self.side_length/1.8
    }

    fn atom_displacement(&self,pos: &Position,max_displacement:NumericData) -> Position {
        let trial_pos = [
            pos[0] + max_displacement*(random::<NumericData>()-0.5),
            pos[1] + max_displacement*(random::<NumericData>()-0.5),
            pos[2] + max_displacement*(random::<NumericData>()-0.5),
        ];
        [
            trial_pos[0] - self.side_length * ((trial_pos[0]-trial_pos[1]/3_f32.sqrt()-self.side_length/2.0)/self.side_length+0.5).floor()
            + self.side_length/2.0*((trial_pos[1]-self.side_length*3_f32.sqrt()/4.0)/(self.side_length*3_f32.sqrt()/2.0)+0.5).floor(),
            trial_pos[1] - self.side_length * 3_f32.sqrt()/2.0 * ((trial_pos[1]-self.side_length*3_f32.sqrt()/4.0)/(self.side_length*3_f32.sqrt()/2.0)+0.5).floor(),
            trial_pos[2] - self.side_height * 3_f32.sqrt()/2.0 * ((trial_pos[2]-self.side_height/2.0)/self.side_height+0.5).floor(),
        ]
    }
}

impl PeriodicTrait for Rhombic {
    fn get_volume(&self) -> NumericData {
        self.side_length * self.side_length * self.side_height * 3_f32.powf(0.5)/2.0
    }

    fn get_r_cut(&self) -> NumericData {
        (self.side_length*self.side_length*3.0/16.0).min(self.side_height*self.side_height/4.0)
    }

    fn get_lrc_scale_factor(&self) -> NumericData {
        0.75 * self.side_length * self.side_height
    }

    fn get_side_length(&self) -> NumericData {
        self.side_height
    }
}

#[enum_dispatch]
pub trait PeriodicTrait: BoundaryConditionTrait {
    fn get_volume(&self) -> NumericData;
    fn get_r_cut(&self) -> NumericData;
    fn get_lrc_scale_factor(&self) -> NumericData {
        1.0
    }
    fn get_side_length(&self) -> NumericData;
}

#[enum_dispatch(PeriodicTrait)]
#[enum_dispatch(BoundaryConditionTrait)]
#[derive(Clone)]
pub enum Periodic {
    Cubic(Cubic),
    Rhombic(Rhombic),
}
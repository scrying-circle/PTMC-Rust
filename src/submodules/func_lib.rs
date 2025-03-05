use crate::submodules::type_lib::*;

pub fn distance_squared(pos1: &Position, pos2: &Position) -> NumericData {
    pos1.iter().zip(pos2.iter()).fold(0.0, |acc, (x, y)| acc + (x-y)*(x-y))
}

pub fn get_tan(pos1: &Position, pos2: &Position) -> NumericData {
    ((pos1[0]-pos2[0])*(pos1[0]-pos2[0]) + (pos1[1]-pos2[1])*(pos1[1]-pos2[1])).sqrt()/(pos1[2]-pos2[2])
}
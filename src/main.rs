use scripts::{ne13, ne55};
use submodules::{input_params::{MCParams, Output}, system_solver::SystemSolverField, test_contents::TestContents};

mod scripts;
mod submodules;
fn main() {
    let results = ne55::run();
    results.plot(&SystemSolverField::T, &SystemSolverField::C_v).unwrap();
    results.plot(&SystemSolverField::T, &SystemSolverField::dC_v).unwrap();
}


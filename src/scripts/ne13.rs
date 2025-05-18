use ndarray::Array;

use crate::submodules::{boundary_conditions::Spherical, configurations::Configuration, energy_evaluation::{ELJEven, PotentialKinds}, ensembles::{EnsembleTrait, NVT}, input_params::{MCParams, TempGrid}, ptmc::PTMC, system_solver::SystemSolver, type_lib::NumericData};

pub fn run() -> SystemSolver {
    let n_atoms = 13;

    let ti = 4.0;
    let tf = 16.0;
    let n_traj = 25;

    let temp_grid = TempGrid::new_equally_spaced(ti, tf, n_traj);

    let mc_cycles = 1000;
    let mc_sample = 1;

    let atom_displacement = 0.1;
    let adjust_period = 100;

    let max_atom_displacement = (0..n_traj).map(|i| 0.1*(atom_displacement*temp_grid.t_grid[i]).sqrt()).collect::<Vec<_>>();

    let mc_params = MCParams::new(mc_cycles, n_traj, n_atoms, mc_sample, adjust_period);

    let coeffs = vec![-10.5097942564988, 989.725135614556, -101383.865938807, 3918846.12841668, -56234083.4334278, 288738837.441765];
    let potential = PotentialKinds::ELJEven(ELJEven {coeffs});

    let ensemble = NVT::new(n_atoms);
    let move_strategy = ensemble.get_move_strategy();

    let mut position_vector = Array::from_shape_vec(n_atoms, vec![[2.825384495892464, 0.928562467914040, 0.505520149314310],
        [2.023342172678102,	-2.136126268595355, 0.666071287554958],
        [2.033761811732818,	-0.643989413759464, -2.133000349161121],
        [0.979777205108572,	2.312002562803556, -1.671909307631893],
        [0.962914279874254,	-0.102326586625353, 2.857083360096907],
        [0.317957619634043,	2.646768968413408, 1.412132053672896],
        [-2.825388342924982, -0.928563755928189, -0.505520471387560],
        [-0.317955944853142, -2.646769840660271, -1.412131825293682],
        [-0.979776174195320, -2.312003751825495, 1.671909138648006],
        [-0.962916072888105, 0.102326392265998,	-2.857083272537599],
        [-2.023340541398004, 2.136128558801072,	-0.666071089291685],
        [-2.033762834001679, 0.643989905095452, 2.132999911364582],
        [0.000002325340981,	0.000000762100600, 0.000000414930733]]).unwrap();

    const A_TO_BOHR: NumericData = 1.8897259886;
    for pos in position_vector.iter_mut() {
        pos[0] *= A_TO_BOHR;
        pos[1] *= A_TO_BOHR;
        pos[2] *= A_TO_BOHR;
    }

    let bc = Spherical {radius2: 5.32*A_TO_BOHR * 5.32*A_TO_BOHR};

    let start_config = Configuration::<Spherical>::new_spherical(position_vector, bc);

    let mut ptmc = PTMC::run(mc_params, temp_grid, ensemble, potential, start_config, false);
    ptmc.postprocess()
}
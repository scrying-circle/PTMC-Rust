use std::mem::MaybeUninit;

use ndarray::{Array, Array1, Array2, ArrayBase, Axis, Dim, OwnedRepr};
use ndarray_linalg::Solve;

use super::type_lib::NumericData;

pub struct SystemSolver{
    pub z: Array1<NumericData>,
    pub u: Array1<NumericData>,
    pub c_v: Array1<NumericData>,
    pub dc_v: Array1<NumericData>,
    pub s: Array1<NumericData>,
    pub t: Array1<NumericData>,
}

impl SystemSolver {
    pub fn multihistogram(histogram: &Array2<NumericData>, energy_vector: &Vec<NumericData>, beta: &Vec<NumericData>, sums: &Vec<NumericData>) -> Self {
        let (alpha, s) = SystemSolver::solve(histogram, energy_vector, beta, sums);
        SystemSolver::analysis(energy_vector, s, beta, 1000)
    }

    fn analysis(energy_vector: &Vec<NumericData>, s: Array1<NumericData>, beta: &Vec<NumericData>, n_points: usize) -> Self {
        const K_B: NumericData = 3.16681196e-6;

        let n_bins = energy_vector.len();
        let temp_vec = Array::from_shape_fn(beta.len(), |i| 1.0/(K_B*beta[i]));
        let delta_temp = (temp_vec.last().unwrap() - temp_vec.first().unwrap())/n_points as NumericData;
        let t_scale = (0..n_points).map(|i| temp_vec.first().unwrap() + i as NumericData * delta_temp).collect::<Array1<_>>();

        let mut n_exp = 0.0;
        let y = Array::from_shape_fn((n_points, n_bins), |(i, j)| {
            s[j] - energy_vector[j]/(t_scale[i]*K_B)
        });
        let mut xp: ArrayBase<OwnedRepr<MaybeUninit<NumericData>>, Dim<[usize; 2]>> = Array::uninit((n_points, n_bins));
        let mut z: ArrayBase<OwnedRepr<MaybeUninit<NumericData>>, Dim<[usize; 1]>> = Array::uninit(n_points);

        for i in 0..n_points {
            loop {
                let mut sum = 0.0;
                for j in 0..n_bins {
                    let value = (y[ [i, j] ] - n_exp).exp();
                    sum += value;
                    xp[ [i, j] ] = MaybeUninit::new(value);
                }
                z[i] = MaybeUninit::new(sum);

                if sum < 1.0 {
                    n_exp -= 1.0;
                } else if sum > 100.0 {
                    n_exp += 0.7;
                } else {
                    break
                }
            }
        }
        let safe_z;
        let safe_xp;
        unsafe { safe_z = z.assume_init(); safe_xp = xp.assume_init(); };

        let u = Array::from_shape_fn(n_points, |i| {
            let mut sum = 0.0;
            for j in 0..n_bins {
                sum += safe_xp[ [i, j] ] * energy_vector[j]
            }
            sum/safe_z[i]
        });

        let u2 = Array::from_shape_fn(n_points, |i| {
            let mut sum = 0.0;
            for j in 0..n_bins {
                sum += safe_xp[ [i, j] ] * energy_vector[j] * energy_vector[j]
            }
            sum/safe_z[i]
        });

        let r2 = Array::from_shape_fn(n_points, |i| {
            let mut sum = 0.0;
            for j in 0..n_bins {
                sum += safe_xp[ [i, j] ] * (energy_vector[j] - u[i]) * (energy_vector[j] - u[i])
            }
            sum/safe_z[i]
        });
        
        let r3 = Array::from_shape_fn(n_points, |i| {
            let mut sum = 0.0;
            for j in 0..n_bins {
                sum += safe_xp[ [i, j] ] * (energy_vector[j] - u[i]) * (energy_vector[j] - u[i])* (energy_vector[j] - u[i])
            }
            sum/safe_z[i]
        });

        let c_v = Array::from_shape_fn(n_points, |i| {
            (u2[i] - u[i]*u[i])/(t_scale[i]*t_scale[i])/K_B
        });

        let s_t = Array::from_shape_fn(n_points, |i| {
            u[i]/t_scale[i] + K_B * safe_z[i].ln()
        });

        let dc_v = Array::from_shape_fn(n_points, |i| {
            r3[i]/(K_B * K_B * t_scale[i].powi(4)) - 2.0 * r2[i]/(K_B * t_scale[i].powi(3))
        });

        SystemSolver {
            z: safe_z,
            u,
            c_v,
            dc_v,
            s: s_t,
            t: t_scale,
        }
    }

    fn solve(histogram: &Array2<NumericData>, energy_vector: &Vec<NumericData>, beta: &Vec<NumericData>, sums: &Vec<NumericData>) -> (Array1<NumericData>, Array1<NumericData>) {
        let (b, bmat) = SystemSolver::bvector(histogram, energy_vector, beta, sums);
        let a = SystemSolver::amatrix(histogram, &sums);

        let alpha = a.solve(&b).expect("matrix solve failed");
        let s = SystemSolver::get_entropy(&alpha, bmat, histogram, sums);
        (alpha, s)
    }

    fn get_entropy(alpha: &Array1<NumericData>, bmat: Array2<NumericData>, histogram: &Array2<NumericData>, sums: &Vec<NumericData>) -> Array1<NumericData>{
        let (_n_traj, n_bins) = SystemSolver::get_dims(histogram);
        Array::from_shape_fn(n_bins, |i| {
            Array::from_shape_fn(n_bins, |j| bmat[ [i, j] ] - histogram[ [i, j] ] * alpha[j]).sum()/sums[i]
        })
    }

    fn get_dims(histogram: &Array2<NumericData>) -> (usize, usize) {
        (histogram.shape()[0], histogram.shape()[1])
    }

    fn bvector(histogram: &Array2<NumericData>, energy_vector: &Vec<NumericData>, beta: &Vec<NumericData>, sums: &Vec<NumericData>) -> (Array1<NumericData>, Array2<NumericData>) {
        let (n_traj, n_bins) = SystemSolver::get_dims(histogram);
        let logmat = Array::from_shape_fn((n_traj, n_bins), |(i, j)| histogram[[i, j]].ln() + beta[i]*energy_vector[j]);

        let bmat = Array::from_shape_fn((n_traj, n_bins), |(i, j)| if histogram[[i, j]] == 0.0 {0.0} else {histogram[[i, j]]*logmat[[i, j]]});
        let rh_vec = bmat.axis_iter(Axis(1)).map(|row| row.sum()).collect::<Vec<_>>();

        let b = bmat.axis_iter(Axis(0)).enumerate()
        .map(|(count, row)| row.sum() - Array::from_shape_fn(n_bins, |i| histogram[ [count, i] ] * rh_vec[i]/sums[i]).sum()).collect::<Array1<_>>();
        
        (b, bmat)
    }

    fn amatrix(histogram: &Array2<NumericData>, sums: &Vec<NumericData>) -> Array2<NumericData> {
        let (n_traj, _n_bins) = SystemSolver::get_dims(histogram);
        Array::from_shape_fn((n_traj,n_traj), |(i, j)| {
            let total = Array::from_shape_fn(n_traj, |k| histogram[ [i, k] ]*histogram[ [j, k] ]/sums[k]).sum() * -1.0;
            if i == j {total + histogram.row(i).sum()} else {total}
        })
    }
}
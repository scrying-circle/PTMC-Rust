use std::{fmt::format, io::{BufRead, Write}};

use ndarray::{Array, Array1, Array2};
use serde::Serialize;

use crate::submodules::energy_evaluation;

use super::{input_params::EnergyBounds, type_lib::NumericData};

pub struct TestContents {
    pub raw_hist: Array2<NumericData>,
    pub energy_bounds: EnergyBounds,
    pub s: Array1<NumericData>,
    pub bmat: Array2<NumericData>,
    pub b: Array1<NumericData>,
    pub alpha: Array1<NumericData>,
    pub sums: Vec<NumericData>,
    pub processed_histogram: Array2<NumericData>,
    pub energy_vector: Vec<NumericData>,
}

impl TestContents {
    pub fn readfiles() -> Self {
        const INPUT_FOLDER: &str = "input_files/";
        let filenames = vec![
            "raw_hist.json",
            "energy_bounds.json",
            "s.json",
            "bmat.json",
            "b.json",
            "alpha.json",
            "sums.json",
            "processed_histogram.json",
            "energy_vector.json",
        ];
        let file = std::fs::read_to_string(format!("{}{}", INPUT_FOLDER, filenames[0])).unwrap();
        let raw_hist: Array2<NumericData> = serde_json::from_str(&file).unwrap();

        let file = std::fs::read_to_string(format!("{}{}", INPUT_FOLDER, filenames[1])).unwrap();
        let energy_bounds: EnergyBounds = serde_json::from_str(&file).unwrap();

        let file = std::fs::read_to_string(format!("{}{}", INPUT_FOLDER, filenames[2])).unwrap();
        let s: Array1<NumericData> = serde_json::from_str(&file).unwrap();

        let file = std::fs::read_to_string(format!("{}{}", INPUT_FOLDER, filenames[3])).unwrap();
        let bmat: Array2<NumericData> = serde_json::from_str(&file).unwrap();

        let file = std::fs::read_to_string(format!("{}{}", INPUT_FOLDER, filenames[4])).unwrap();
        let b: Array1<NumericData> = serde_json::from_str(&file).unwrap();

        let file = std::fs::read_to_string(format!("{}{}", INPUT_FOLDER, filenames[5])).unwrap();
        let alpha: Array1<NumericData> = serde_json::from_str(&file).unwrap();

        let file = std::fs::read_to_string(format!("{}{}", INPUT_FOLDER, filenames[6])).unwrap();
        let sums: Vec<NumericData> = serde_json::from_str(&file).unwrap();

        let file = std::fs::read_to_string(format!("{}{}", INPUT_FOLDER, filenames[7])).unwrap();
        let processed_histogram: Array2<NumericData> = serde_json::from_str(&file).unwrap();

        let file = std::fs::read_to_string(format!("{}{}", INPUT_FOLDER, filenames[8])).unwrap();
        let energy_vector: Vec<NumericData> = serde_json::from_str(&file).unwrap();

        TestContents {raw_hist, energy_bounds, s, bmat, b, alpha, sums, processed_histogram, energy_vector}
    }

    pub fn print<T: Serialize + ?Sized>(item: &T, filename: &str) {
        let real_filename = format!("output_files/{}", filename);
        let mut file = std::fs::File::create(real_filename).unwrap();
        let json = serde_json::to_string(&item).unwrap();
        file.write_all(json.as_bytes()).unwrap();
    }
}
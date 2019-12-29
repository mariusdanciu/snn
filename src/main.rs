extern crate rand;
extern crate rand_chacha;

use std::time::{SystemTime, UNIX_EPOCH};

use nalgebra::*;
use rand::{Rng, SeedableRng};

use crate::neunet::definitions::MLOps;
use crate::neunet::files::idx::IdxFile;
use crate::neunet::loader::DataLoader;
use crate::neunet::utils::matrix::MatrixUtil;
use std::ops::IndexMut;
mod neunet;

fn main() {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(10);
    let vec = DVector::<f64>::from_fn(5, |_, _| rng.gen::<f64>());

    println!("{}", vec);

    println!("Random f32: {}", rng.gen::<f64>());

    let rand_epsilon = 0.03_f64;

    let m = DMatrix::<f64>::from_fn(4, 4, |_, _| rng.gen::<f64>() * 2. * rand_epsilon - rand_epsilon);

    println!("{}", m);

    let start = SystemTime::now().duration_since(UNIX_EPOCH)
        .expect("Time went backwards");
    let res = IdxFile.load_data(String::from("./train-images-idx3-ubyte"), String::from("./train-labels-idx1-ubyte"));
    let end = SystemTime::now().duration_since(UNIX_EPOCH)
        .expect("Time went backwards");


    println!("Data read in {} ms", (end - start).as_millis());
    let (img, labels) = res.unwrap();

    println!("Shapes {:?}", img.shape());
    let r = img.column(30);
    let label = labels[30];


    println!("Img {}", DMatrix::from_row_slice(28, 28, r.as_slice()));
    println!("Label {}", label);


    let b = DVector::from_vec(vec![11.0, 12.0, 13.0]);
    let x = DVector::from_vec(vec![1, 2, 3]);

    let mut w = DMatrix::from_vec(3, 3, vec![1.0, 2.0, 3.0, 4., 5.0, 6.0, 7., 8., 9.]);


    println!("w {}", w);

    MatrixUtil::set_row(&mut w, 2, &b);
    println!("w {}", w);



}
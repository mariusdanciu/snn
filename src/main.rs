extern crate rand;
extern crate rand_chacha;

use std::time::{SystemTime, UNIX_EPOCH};

use nalgebra::*;
use rand::{Rng, SeedableRng};

use crate::neunet::definitions::MLOps;
use crate::neunet::files::idx::IdxFile;
use crate::neunet::loader::DataLoader;

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


    let b = DVector::from_vec(vec![1, 2]);
    let x = DVector::from_vec(vec![1, 2, 3]);
    let w = DMatrix::from_vec(2, 3, vec![1, 2, 3, 4, 5, 6]);
    println!("w {}", w);
    println!("x {}", x);
    println!("b {}", b);

    println!("z {:.5}", (w * x + b).map(|e| {
        MLOps::sigmoid(e as f64)
    }));

}

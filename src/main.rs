extern crate rand;
extern crate rand_chacha;

use std::time::{SystemTime, UNIX_EPOCH};

use nalgebra::*;
use rand::{Rng, SeedableRng};

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
    let res = IdxFile.load_data(String::from("./train-images-idx3-ubyte"));
    let end = SystemTime::now().duration_since(UNIX_EPOCH)
        .expect("Time went backwards");


    println!("Data read in {} ms", (end - start).as_millis());
    let img = res.unwrap();

    println!("Shapes {:?}", img.shape());
    let r = img.column(30);

    println!("Img {}", DMatrix::from_row_slice(28, 28, r.as_slice()));

}

#[macro_use]
extern crate ndarray;

mod neunet;

use crate::neunet::files::idx::IdxFile;
use crate::neunet::loader::DataLoader;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

fn main() {

    let start = SystemTime::now().duration_since(UNIX_EPOCH)
        .expect("Time went backwards");
    let res = IdxFile.load_data(String::from("./train-images-idx3-ubyte"));
    let end = SystemTime::now().duration_since(UNIX_EPOCH)
        .expect("Time went backwards");;

    println!("Data read in {} ms", (end - start).as_millis());
    let img = res.unwrap();

    println!("Res = {:3?}", img.slice(s![60..61, 0..]).into_shape((28, 28)));
}

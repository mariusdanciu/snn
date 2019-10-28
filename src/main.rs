#[macro_use]
extern crate ndarray;

mod neunet;

use crate::neunet::files::idx::IdxFile;
use crate::neunet::loader::DataLoader;

fn main() {

    let res = IdxFile.load_data(String::from("./train-images-idx3-ubyte"));

    let img = res.unwrap();

    println!("Res = {:3?}", img.slice(s![60..61, 0..]).into_shape((28, 28)));
}

use std::fs::File;
use std::io::*;

use ndarray::*;

use crate::neunet::loader::DataLoader;
use crate::neunet::utils::binary::BinaryOps;

pub struct IdxFile;

impl DataLoader for IdxFile {
    fn load_data(self, path: String) -> Result<Array2<u8>> {
        println!("Path {}", path);
        let mut buf: [u8; 8] = [0; 8];

        let f = File::open(path)?;
        let mut reader = BufReader::new(f);

        reader.read(&mut buf)?;

        let magic = BinaryOps::to_u32(&buf[0..4]);
        let num = BinaryOps::to_u32(&buf[4..8]) as usize;

        reader.read(&mut buf)?;

        let x_size = BinaryOps::to_u32(&buf[0..4]);
        let y_size = BinaryOps::to_u32(&buf[4..8]);
        let img_size = (x_size * y_size) as usize;


        println!("Magic {}", magic);
        println!("Num images: {}", num);
        println!("X size: {}", x_size);
        println!("Y size: {}", y_size);
        println!("Img size {}", img_size);

        let mut out: Vec<u8> = vec![0; 0];

        let read_bytes = reader.read_to_end(&mut out)?;

        println!("read_bytes {}", read_bytes);
        let r = Array::from_shape_vec((num, img_size), out);
        Ok(r.unwrap())
    }
}

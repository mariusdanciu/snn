use std::fs::File;
use std::io::*;

use nalgebra::*;

use crate::neunet::loader::DataLoader;
use crate::neunet::utils::binary::BinaryOps;

pub struct IdxFile;

impl IdxFile {
    fn read_head(reader: &mut BufReader<File>) -> Result<u32> {
        let mut buf: [u8; 8] = [0; 8];
        reader.read(&mut buf)?;
        let magic = BinaryOps::to_u32(&buf[0..4]);
        Ok(BinaryOps::to_u32(&buf[4..8]))
    }

    fn read_data(data_path: String) -> Result<DMatrix<u8>> {
        let f = File::open(data_path)?;
        let mut buf: [u8; 8] = [0; 8];
        let mut data_reader = BufReader::new(f);

        let num = IdxFile::read_head(&mut data_reader).unwrap() as usize;

        data_reader.read(&mut buf)?;

        let x_size = BinaryOps::to_u32(&buf[0..4]);
        let y_size = BinaryOps::to_u32(&buf[4..8]);
        let img_size = (x_size * y_size) as usize;


        println!("Num images: {}", num);
        println!("X size: {}", x_size);
        println!("Y size: {}", y_size);
        println!("Img size {}", img_size);

        let mut out: Vec<u8> = vec![0; 0];

        let read_bytes = data_reader.read_to_end(&mut out)?;

        println!("read_bytes {}", read_bytes);
        let r = DMatrix::from_vec(img_size, num, out);
        Ok(r)
    }

    fn read_labels(labels_path: String) -> Result<DVector<u8>> {
        let f = File::open(labels_path)?;
        let mut buf: [u8; 8] = [0; 8];
        let mut data_reader = BufReader::new(f);

        let num = IdxFile::read_head(&mut data_reader).unwrap() as usize;

        println!("Num images: {}", num);

        let mut out: Vec<u8> = vec![0; 0];

        let read_bytes = data_reader.read_to_end(&mut out)?;

        let r = DVector::from_vec(out);
        Ok(r)
    }
}


impl DataLoader for IdxFile {
    fn load_data(self, data_path: String, labels_path: String) -> Result<(DMatrix<u8>, DVector<u8>)> {
        let data = IdxFile::read_data(data_path)?;
        let labels = IdxFile::read_labels(labels_path)?;
        Ok((data, labels))
    }
}

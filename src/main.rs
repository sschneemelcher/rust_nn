extern crate nalgebra as na;
extern crate rand;

use csv::StringRecord;
use na::SMatrix;
use rand::{thread_rng, Rng};
use rand::distributions::Uniform;

use std::env;
use std::error::Error;
use std::ffi::OsString;
use std::fs::File;
use std::process;


type Matrix1x784f   = SMatrix<f32, 1, 784>;
type Matrix784x128f = SMatrix<f32, 784, 128>;
type Matrix128x64f  = SMatrix<f32, 128, 64>;
type Matrix64x1f    = SMatrix<f32, 64, 1>;
type Matrix1x128f    = SMatrix<f32,1, 128>;
type Matrix1x64f    = SMatrix<f32, 1, 64>;
type Record         = (i32, Vec<f32>);


fn sigmoid(x: f32) -> f32 {
  if x < -25.0 {
      0.0
  } else if x > 25.0 {
      1.0
  } else {
    1.0 / (1.0 + f32::exp(-x))
  }
}

struct NeuralNetwork {
    layer1: Matrix784x128f,
    layer2: Matrix128x64f,
    layer3: Matrix64x1f,
    bias1:  Matrix1x128f,
    bias2:  Matrix1x64f,
}

impl NeuralNetwork {

    fn predict(&mut self, input: Matrix1x784f) -> f32 {

        let mut l1 = input * self.layer1 + self.bias1;
        l1.apply(|x| {sigmoid(x)});
        let mut l2 = l1 * self.layer2 + self.bias2;
        l2.apply(|x| {sigmoid(x)});
        let mut l3 = l2 * self.layer3;
        l3.apply(|x| {sigmoid(x)});
        return l3.sum(); 
    }

}

fn create_net() -> NeuralNetwork {
        let mut rng = thread_rng();

        let mut nn = NeuralNetwork {
        layer1: Matrix784x128f::from_iterator((&mut rng).sample_iter(Uniform::from(-0.5..0.5))),
        layer2: Matrix128x64f::from_iterator((&mut rng).sample_iter(Uniform::from(-0.5..0.5))),
        layer3: Matrix64x1f::from_iterator((&mut rng).sample_iter(Uniform::from(-0.5..0.5))),
        bias1:  Matrix1x128f::from_iterator((&mut rng).sample_iter(Uniform::from(-0.5..0.5))),
        bias2:  Matrix1x64f::from_iterator((&mut rng).sample_iter(Uniform::from(-0.5..0.5))),
        };
        return nn;
}


fn load_data() -> Result<(Vec<Matrix1x784f>, Vec<i32>), Box<dyn Error>> {
    let file_path = get_arg()?;
    let file = File::open(file_path)?;
    let mut rdr = csv::Reader::from_reader(file);
    let mut x_train: Vec<Matrix1x784f> = [].to_vec();
    let mut y_train: Vec<i32> = [].to_vec();
    for result in rdr.deserialize() {
        let record: Record = result?;
        y_train.push(record.0);
        x_train.push(Matrix1x784f::from_vec(record.1));
    }
    Ok((x_train, y_train))
}

fn get_arg() -> Result<OsString, Box<dyn Error>> {
    match env::args_os().nth(1) {
        None => Err(From::from("expected an argument")),
        Some(file_path) => Ok(file_path),
    }
}

fn main() {
    match load_data() {
        Err(err) => {println!("{}", err);
                    process::exit(1);},
        Ok((x_train, y_train)) => {
    
            let mut rng = thread_rng();
            let mut nn = create_net();
            println!("{:#?}", nn.predict(x_train[0]));
        }
    }
}

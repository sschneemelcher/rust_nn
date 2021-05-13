extern crate nalgebra as na;
extern crate rand;

use na::SMatrix;
use rand::{thread_rng, Rng};
use rand::distributions::Uniform;


type Matrix1x310f = SMatrix<f32, 1, 310>;
type Matrix310x128f = SMatrix<f32, 310, 128>;
type Matrix128x64f  = SMatrix<f32, 128, 64>;
type Matrix64x1f    = SMatrix<f32, 64, 1>;


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
    layer1: Matrix310x128f,
    layer2: Matrix128x64f,
    layer3: Matrix64x1f,
    bias1: f32,
    bias2: f32,
}

impl NeuralNetwork {

    fn predict(&mut self, input: Matrix1x310f) -> f32 {

        let mut l1 = input * self.layer1;
        l1.apply(|x| {sigmoid(x + self.bias1)});
        let mut l2 = l1 * self.layer2;
        l2.apply(|x| {sigmoid(x + self.bias2)});
        let mut l3 = l2 * self.layer3;
        l3.apply(|x| {sigmoid(x)});
        return l3.sum(); 
    }

}

fn CreateNet() -> NeuralNetwork {
        let mut rng = thread_rng();

        let mut nn = NeuralNetwork {
        layer1: Matrix310x128f::from_iterator((&mut rng).sample_iter(Uniform::from(-0.5..0.5))),
        layer2: Matrix128x64f::from_iterator((&mut rng).sample_iter(Uniform::from(-0.5..0.5))),
        layer3: Matrix64x1f::from_iterator((&mut rng).sample_iter(Uniform::from(-0.5..0.5))),
        bias1: rng.gen(),
        bias2: rng.gen()};
        return nn;
}


fn main() {
    let mut rng = thread_rng();
    let mut x_train = Matrix1x310f::from_iterator((&mut rng).sample_iter(Uniform::from(-0.5..0.5)));
    let mut nn = CreateNet();
    println!("{:#?}", nn.predict(x_train));
}

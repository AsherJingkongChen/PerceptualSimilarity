//! LPIPS with the AlexNet trunk, on Burn: the first step of the port, to be matched against the Python package.
//!
//! Usage: lpips WEIGHTS.safetensors original.png modified.png [modified.png...]
//!
//! WEIGHTS is written by scripts/export.py from the Python package. Prints one line per modified image: its
//! distance to the original, a tab, its path; the time each took goes to stderr.

use std::{env, fs, process::ExitCode, time::Instant};

use burn::tensor::{
    Device, Tensor, TensorData,
    activation::relu,
    module::{conv2d, max_pool2d},
    ops::{ConvOptions, MaxPoolOptions},
};
use safetensors::SafeTensors;

/// One convolution of AlexNet's features: its weights, stride and padding, and whether a 3 x 3 max pool with
/// stride 2 comes before it.
struct Conv {
    weight: Tensor<4>,
    bias: Tensor<1>,
    stride: usize,
    padding: usize,
    pool_before: bool,
}

/// The network: the input's shift and scale, the five convolutions whose ReLUs are compared, and the five 1 x 1
/// linear layers that weigh each channel's difference.
struct Lpips {
    shift: Tensor<4>,
    scale: Tensor<4>,
    convs: Vec<Conv>,
    lins: Vec<Tensor<4>>,
}

fn tensor<const D: usize>(st: &SafeTensors, name: &str, device: &Device) -> Result<Tensor<D>, String> {
    let view = st.tensor(name).map_err(|e| format!("{name}: {e}"))?;
    let values: Vec<f32> = view
        .data()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    Ok(Tensor::from_data(TensorData::new(values, view.shape().to_vec()), device))
}

impl Lpips {
    fn load(path: &str, device: &Device) -> Result<Self, String> {
        let bytes = fs::read(path).map_err(|e| format!("{path}: {e}"))?;
        let st = SafeTensors::deserialize(&bytes).map_err(|e| format!("{path}: {e}"))?;
        // (where it sits in torchvision's features, stride, padding, a pool before it)
        let layout = [("slice1.0", 4, 2, false), ("slice2.3", 1, 2, true), ("slice3.6", 1, 1, true),
                      ("slice4.8", 1, 1, false), ("slice5.10", 1, 1, false)];
        let mut convs = Vec::new();
        for (name, stride, padding, pool_before) in layout {
            convs.push(Conv {
                weight: tensor(&st, &format!("net.{name}.weight"), device)?,
                bias: tensor(&st, &format!("net.{name}.bias"), device)?,
                stride,
                padding,
                pool_before,
            });
        }
        let mut lins = Vec::new();
        for k in 0..5 {
            lins.push(tensor(&st, &format!("lin{k}.model.1.weight"), device)?);
        }
        Ok(Self {
            shift: tensor(&st, "scaling_layer.shift", device)?,
            scale: tensor(&st, "scaling_layer.scale", device)?,
            convs,
            lins,
        })
    }

    /// The distance between two batches of images in [-1, 1], one number per pair.
    fn distance(&self, x: Tensor<4>, y: Tensor<4>) -> Tensor<1> {
        let n = x.dims()[0];
        let mut h = (Tensor::cat(vec![x, y], 0) - self.shift.clone()) / self.scale.clone();
        let mut total: Option<Tensor<4>> = None;
        for (conv, lin) in self.convs.iter().zip(&self.lins) {
            if conv.pool_before {
                h = max_pool2d(h, MaxPoolOptions::new([3, 3]).with_stride([2, 2]));
            }
            let options = ConvOptions::new([conv.stride; 2], [conv.padding; 2], [1, 1], 1);
            h = relu(conv2d(h, conv.weight.clone(), Some(conv.bias.clone()), options));
            // each feature divided by its length over the channels, as lpips.normalize_tensor does
            let unit = h.clone() / (h.clone().square().sum_dim(1).sqrt() + 1e-10);
            let diff = (unit.clone().narrow(0, 0, n) - unit.narrow(0, n, n)).square();
            let layer = conv2d(diff, lin.clone(), None, ConvOptions::new([1, 1], [0, 0], [1, 1], 1))
                .mean_dim(2)
                .mean_dim(3);
            total = Some(match total {
                Some(t) => t + layer,
                None => layer,
            });
        }
        match total {
            Some(t) => t.reshape([n]),
            None => Tensor::zeros([n], &self.shift.device()),
        }
    }
}

/// An image as a 1 x 3 x H x W tensor in [-1, 1], as lpips.im2tensor makes it from 8-bit RGB.
fn read(path: &str, device: &Device) -> Result<Tensor<4>, String> {
    let img = image::open(path).map_err(|e| format!("{path}: {e}"))?.to_rgb8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let mut chw = vec![0f32; 3 * w * h];
    for (i, p) in img.pixels().enumerate() {
        for c in 0..3 {
            chw[c * w * h + i] = f32::from(p[c]) / 127.5 - 1.0;
        }
    }
    Ok(Tensor::from_data(TensorData::new(chw, [1, 3, h, w]), device))
}

fn run(args: &[String]) -> Result<(), String> {
    let [weights, original, modified @ ..] = args else {
        return Err("usage: lpips WEIGHTS.safetensors original.png modified.png [modified.png...]".into());
    };
    if modified.is_empty() {
        return Err("give at least one modified image".into());
    }
    let device = Device::default();
    let net = Lpips::load(weights, &device)?;
    let x = read(original, &device)?;
    for path in modified {
        let y = read(path, &device)?;
        if y.dims() != x.dims() {
            return Err(format!("{path} is {:?}, {original} is {:?}", y.dims(), x.dims()));
        }
        let t0 = Instant::now();
        let d: f32 = net.distance(x.clone(), y).into_scalar();
        eprintln!("{} ms", t0.elapsed().as_millis());
        println!("{d:.10}\t{path}");
    }
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("lpips: {e}");
            ExitCode::FAILURE
        }
    }
}

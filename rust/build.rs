//! Grows the network out of model/lpips_alex.onnx, which scripts/export_onnx.py writes from the Python package.

use burn_onnx::ModelGen;

fn main() {
    ModelGen::new().input("model/lpips_alex.onnx").out_dir("model/").run_from_script();
}

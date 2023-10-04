
use ndarray::{Array, Zip, Ix2, Ix1};

/// Batch Normalization Operator
/// 
/// ONNX definition: https://onnx.ai/onnx/operators/onnx__BatchNormalization.html
///
/// Performs the Batch Normalization Operation
///  
/// - X: (M x N) input
/// - scale: (N x 1) a.k.a gamma
/// - bias: (N x 1) a.k.a beta
/// - Y: (M x N) output
/// - Mode : true = training, false = inference.
/// - Epsilon: The epsilon value used to avoid division by Zero. (a very small float)
/// - Momentum: Factor used to compute the running mean and variance. 
/// 
/// - in_mean: (N x 1) Mean of X on the X-axis.
/// - in_var: (N x 1) Variance of X on the X-axis.
/// 
/// - running_mean: 
pub fn batch_norm(
    x: &Array<f32, Ix2>,
    scale: &Array<f32, Ix1>,
    bias: &Array<f32, Ix1>,
    y: &mut Array<f32, Ix2>,
    mode: bool,
    epsilon: f32,
    momentum: f32,

    in_mean: &Array<f32, Ix1>,
    in_var : &Array<f32, Ix1>,

    running_mean: &mut Array<f32, Ix1>,
    running_var : &mut Array<f32, Ix1>,
) {
    if mode {
        let (m, n) = x.dim();


    }

    todo!()
}

pub fn batch_norm_wrt_a(

) {
    todo!()
}

use ndarray::{Array, Dimension, Zip};

/// Sin Operator
/// 
/// ONNX definition: https://onnx.ai/onnx/operators/onnx__Cos.html#cos
/// 
/// Performs the operation b = cos(a)
/// 
/// - A: Input
/// - B: Output 
pub fn cos<D>(
    a: &Array<f32, D>,
    b: &mut Array<f32, D>,
) 
where
    D: Dimension,
{
    Zip::from(b).and(a).for_each(|b, a| {
        *b = a.cos()
    })
}

/// Cos Operator w.r.t. A
/// 
/// Performs the operation g *= sin(a)
/// 
/// - A: Input
/// - B: Gradient
/// 
/// A _must_ be the A used in the forward operation. 
pub fn cos_wrt_a<D>(
    a: &Array<f32, D>,
    g: &mut Array<f32, D>,
)
where
    D: Dimension,
{
    Zip::from(g).and(a).for_each(|g, a| {
        *g *= a.sin()
    })
}
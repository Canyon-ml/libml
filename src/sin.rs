
use ndarray::{Array, Dimension, Zip};

/// Sin Operator
/// 
/// ONNX definition: https://onnx.ai/onnx/operators/onnx__Sin.html#sin
/// 
/// Performs the operation b = sin(a)
/// 
/// - A: Input
/// - B: Output 
pub fn sin<D>(
    a: &Array<f32, D>,
    b: &mut Array<f32, D>,
) 
where
    D: Dimension,
{
    Zip::from(b).and(a).for_each(|b, a| {
        *b = a.sin()
    })
}

/// Sin Operator w.r.t. A
/// 
/// Performs the operation g *= cos(a)
/// 
/// - A: Input
/// - B: Gradient
/// 
/// A _must_ be the A used in the forward operation. 
pub fn sin_wrt_a<D>(
    a: &Array<f32, D>,
    g: &mut Array<f32, D>,
)
where
    D: Dimension,
{
    Zip::from(g).and(a).for_each(|g, a| {
        *g *= a.cos()
    })
}
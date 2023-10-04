
use ndarray::{Array, Dimension, Zip};

/// # Relu Operator
/// 
/// ONNX definition: https://onnx.ai/onnx/operators/onnx__Relu.html#relu
/// 
/// ## Summary
/// 
/// Performs the element-wise operation f(x) = max(0, x)
/// 
/// ## Inputs
/// 
/// - A: Input 
/// 
/// ## Outputs
/// 
/// - B: Output 
#[inline]
pub fn relu<D>(
    a: &Array<f32, D>, 
    b: &mut Array<f32, D>,
) 
where
    D: Dimension,
{
    Zip::from(b).and(a).for_each(|b, a| {
        *b = if *a > 0. { *a } else { 0. }
    })
}

/// # Relu Gradient w.r.t. A
/// 
/// Performs the operation Gi *= if Bi > 0 { 1 } else { 0 }
/// 
/// - B: Input Array
/// - G: Gradient
/// 
/// B _must_ be the same B used in the forward op.
#[inline]
pub fn relu_wrt_a<D>(
    b: &Array<f32, D>, 
    g: &mut Array<f32, D>,
) 
where
    D: Dimension,
{
    Zip::from(g).and(b).for_each(|g, b| {
        if *b <= 0. {
            *g = 0.
        }
    })
}

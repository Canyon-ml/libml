
use ndarray::{Array, Dimension, Zip};

/// # Sigmoid Operator
/// 
/// ONNX definition: https://onnx.ai/onnx/operators/onnx__Sigmoid.html#sigmoid
/// 
/// Computes Bi = 1. / (1. + e^-Ai).
/// 
/// - A: The Input Array
/// - B: The Output Array
#[inline]
pub fn sigmoid<D>(
    a: &Array<f32, D>,
    b: &mut Array<f32, D>,
) 
where
    D: Dimension,
{
    Zip::from(b).and(a).for_each(|b, a| {
        *b = 1. / (1. + f32::exp(-a))
    })
}

/// # Sigmoid Function Backwards
/// 
/// Computes Gi *= Bi * (1. - Bi)
/// 
/// - B: The output of the forward sigmoid operation.
/// - G: The output gradient w.r.t. A. 
#[inline]
pub fn sigmoid_wrt_a<D>(
    b: &Array<f32, D>,
    g: &mut Array<f32, D> 
) 
where
    D: Dimension,
{
    Zip::from(g).and(b).for_each(|g, b| {
        *g *= b * (1.- b)
    })
}



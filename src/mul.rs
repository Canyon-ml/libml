
use ndarray::{Array, Dimension, Zip};

/// # Multiplication Operator
/// 
/// ONNX definition: https://onnx.ai/onnx/operators/onnx__Mul.html#mul
/// 
/// ## Summary
/// 
/// Performs element-wise binary multiplication.
/// 
/// ## Inputs
/// 
/// - A: Factor
/// - B: Factor
/// 
/// ## Outputs
/// 
/// - C: Product
#[inline]
pub fn mul<D>(
    a: &Array<f32, D>,
    b: &Array<f32, D>,
    c: &mut Array<f32, D>,
) 
where
    D: Dimension,
{
    Zip::from(c).and(a).and(b).for_each(|c, a, b| {
        *c = a * b
    })
}

/// # Multiplication Gradient w.r.t. A
/// 
/// Performs the Operation G *= B
/// 
/// - B: Factor
/// - G: Gradient
/// 
/// B _must_ be the B used in the forward op.
#[inline]
pub fn mul_wrt_a<D>(
    b: &Array<f32, D>,
    g: &mut Array<f32, D>,
) 
where
    D: Dimension,
{
    Zip::from(g).and(b).for_each(|g, b| {
        *g *= b
    })
}

/// # Multiplication Gradient w.r.t. B
/// 
/// Performs the Operation G *= A
/// 
/// - A: Factor
/// - G: Gradient
/// 
/// A _must_ be the A used in the forward op.
#[inline]
pub fn mul_wrt_b<D>(
    a: &Array<f32, D>,
    g: &mut Array<f32, D>,
) 
where
    D: Dimension,
{
    Zip::from(g).and(a).for_each(|g, a| {
        *g *= a
    })
}


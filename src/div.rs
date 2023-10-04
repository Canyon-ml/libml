
use ndarray::{Array, Dimension, Zip};

/// # Division Operator
/// 
/// ONNX definition: https://onnx.ai/onnx/operators/onnx__Div.html#div
/// 
/// ## Summary
/// 
/// Performs element-wise binary division 
/// 
/// ## Inputs
/// 
/// - A: Dividend
/// - B: Divisor
/// 
/// ## Outputs
/// 
/// - C: Quotient
#[inline]
pub fn div<D>(
    a: &Array<f32, D>,
    b: &Array<f32, D>,
    c: &mut Array<f32, D>,
) 
where
    D: Dimension,
{
    Zip::from(c).and(a).and(b).for_each(|c, a, b| {
        *c = a / b
    })
}

/// # Division Gradient w.r.t. A
/// 
/// Performs the operation G *= A / B^2
/// 
/// - A: Dividend
/// - B: Divisor
/// - G: Gradient
/// 
/// A _must_ be the A used in the forward op.
/// 
/// B _must_ be the B used in the forward op.
#[inline]
pub fn div_wrt_a<D>(
    a: &Array<f32, D>,
    b: &Array<f32, D>,
    g: &mut Array<f32, D>,
) 
where
    D: Dimension,
{
    Zip::from(g).and(a).and(b).for_each(|g, a, b| {
        *g *= a / b.powi(2)
    })
}

/// # Division Gradient w.r.t. B
/// 
/// Performs the operation G *= 1. / A
/// 
/// - A: Dividend
/// - G: Gradient
/// 
/// A _must_ be the A used in the forward op.
#[inline]
pub fn div_wrt_b<D>(
    a: &Array<f32, D>,
    g: &mut Array<f32, D>,
) 
where
    D: Dimension,
{
    Zip::from(g).and(a).for_each(|g, a| {
        *g *= 1. / a
    })  
}
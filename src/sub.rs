
use ndarray::{Array, Dimension, Zip};

/// # Subtraction Operator
/// 
/// ONNX definition: https://onnx.ai/onnx/operators/onnx__Sub.html#sub
/// 
/// Performs the operation C = A - B
/// 
/// - A: Minuend
/// - B: Subtrahend
/// - C: Difference
/// 
/// This operator has a gradient w.r.t. b, but not for a. 
/// See the documentation for more details.
#[inline]
pub fn sub<D>(
    a: &Array<f32, D>,
    b: &Array<f32, D>,
    c: &mut Array<f32, D>,
) 
where
    D: Dimension,
{
    Zip::from(c).and(a).and(b).for_each(|c, a, b| {
        *c = a - b
    })
}

/// # Subtraction Gradient w.r.t. A
/// 
/// - G: Gradient
/// 
/// This function does nothing, since the gradient
/// of the minuend is linear (1). Its' here just to be here.
/// The compiler will remove it anyway
#[inline]
pub fn sub_wrt_a<D>(
    g: &mut Array<f32, D>
) 
where
    D: Dimension,
{
    // do nothing
}

/// # Subtraction Gradient w.r.t. B
/// 
/// Performs the operation G = -G
/// 
/// - G: Gradient
#[inline]
pub fn sub_wrt_b<D>(
    g: &mut Array<f32, D>,
)
where
    D: Dimension
{
    for g in g.iter_mut() {
        *g = -*g
    }
}
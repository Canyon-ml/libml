
use ndarray::{Array, Dimension, Zip};

/// # Add Operator
/// 
/// ONNX Definition: https://onnx.ai/onnx/operators/onnx__Add.html#add
/// 
/// ## Summary
/// 
/// Performs element-wise binary addition.
/// 
/// ## Inputs
/// 
/// - A: First Operand
/// - B: Second Operand
/// 
/// ## Outputs
/// 
/// - C: Result
#[inline]
pub fn add<D>(
    a: &Array<f32, D>,
    b: &Array<f32, D>,
    c: &mut Array<f32, D>,
) 
where
    D: Dimension,
{
    Zip::from(c).and(a).and(b).for_each(|c, a, b| {
        *c = a + b
    })
}

/// # Addition Gradient w.r.t. A
/// 
/// - G: Gradient
/// 
/// This function does nothing, since the gradient
/// of addition is linear (1). Its only here to
/// avoid confusion.  See the documentation for more details.
#[inline]
pub fn add_wrt_a<D>(
    _g: &mut Array<f32, D>,
)
where
    D: Dimension,
{
    // do nothing, gradient of addition is linear (1)
}

/// # Addition Gradient w.r.t. B
/// 
/// - G: Gradient
/// 
/// This function does nothing, since the gradient
/// of addition is linear (1). Its only here to
/// avoid confusion.  See the documentation for more details.
#[inline]
pub fn add_wrt_b<D>(
    g: &mut Array<f32, D>,
)
where
    D: Dimension,
{
    // do nothing, gradient of addition is linear (1)
}
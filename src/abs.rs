
use ndarray::{Array, Dimension, Zip};

/// # Absolute Value Operator
/// 
/// ONNX definition: https://onnx.ai/onnx/operators/onnx__Abs.html#l-onnx-doc-abs
/// 
/// ## Summary
/// 
/// Absolute Value takes one input data (Tensor) and produces one 
/// output data (Tensor) where absolute value, y = abs(x), is applied to the tensor elementwise.
/// 
/// ## Inputs
/// 
/// - X: Input Tensor
///  
/// ## Outputs
/// 
/// - Y: Output Tensor
/// 
pub fn abs<D>(
    a: &Array<f32, D>,
    b: &mut Array<f32, D>,
) 
where
    D: Dimension,
{
    Zip::from(b).and(a).for_each(|b, a| {
        *b = a.abs()
    })
}

/// # Absolute Value w.r.t. A
/// 
/// ## Summary
/// 
/// Takes the input tensor of its forward op and multiplies its
/// gradient by G, performing the operation `if a < 0. { g = -g }`.
/// 
/// - A: Input
/// - G: Gradient
/// 
/// A _must_ be the same A used in the forward op. 
pub fn abs_wrt_a<D>(
    a: &Array<f32, D>,
    g: &mut Array<f32, D>,
)
where
    D: Dimension,
{
    Zip::from(g).and(a).for_each(|g, a| {
        if *a < 0. { *g = -*g }
    })
}

use ndarray::{Array, Ix2};

/// # Matrix Multiplication Operator
/// 
/// ONNX Definition: https://onnx.ai/onnx/operators/onnx__MatMul.html#matmul
/// 
/// Performs the operation C = A matmul B.
/// 
/// - A: Input 
/// - B: Input 
/// - C: Output 
/// 
/// Expects A to be mxp, B to  be pxn, and C to be mxn. 
#[inline]
pub fn matmul(
    a: &Array<f32, Ix2>,
    b: &Array<f32, Ix2>,
    c: &mut Array<f32, Ix2>
) {
    // TODO: Find a way to avoid this clone
    a.dot(b).clone_into(c)
}

/// # Matrix Multiplication Gradient w.r.t. A. 
/// 
/// Performs the operation GA = G matmul B^T
/// 
/// - B: Input 
/// - G: Input Gradient
/// - GA: Output Gradient
/// 
/// B _must_ be the same as the B used in the forward op.
#[inline]
pub fn matmul_wrt_a(
    b: &Array<f32, Ix2>,
    g: &Array<f32, Ix2>,
    ga: &mut Array<f32, Ix2>,
) {
    // TODO: Find a way to avoid this clone
    g.dot(&b.t()).clone_into(ga)
}

/// # Matrix Multiplication Gradient w.r.t B. 
/// 
/// Performs the operation GB = G matmul A^T
/// 
/// - A: Input 
/// - G: Input Gradient
/// - GA: Output Gradient
/// 
/// A _must_ be the same as the A used in the forward op.
#[inline]
pub fn matmul_wrt_b(
    a: &Array<f32, Ix2>,
    g: &Array<f32, Ix2>,
    gb: &mut Array<f32, Ix2>,
) {
    // TODO: Find a way to avoid this clone
    g.dot(&a.t()).clone_into(gb)
}
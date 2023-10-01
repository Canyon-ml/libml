
use ndarray::{Array, Ix2, Zip};
use ndarray_stats::QuantileExt;

/// Softmax Operator
/// 
/// ONNX definition: https://onnx.ai/onnx/operators/onnx__Softmax.html#softmax
/// 
/// Performs the operation Bi = e^(Ai) / Sum(j) e^Aj
/// 
/// - A: Input 
/// - B: Output 
pub fn softmax(
    a: &Array<f32, Ix2>,
    b: &mut Array<f32, Ix2>,
) {
    a.clone_into(b);

    for mut row in b.rows_mut() {
        // get and apply the scaling factor.
        // we will use this factor to avoid
        // NaNs, which happens when the
        // inputs are too large or 
        // too small.
        let max = *row.max().unwrap();
        row -= max;

        // Take e^x for each element in the row.
        for b in row.iter_mut() {
            *b = b.exp();
        }

        // get the sum of all the elements in the row.
        let sum = row.sum();

        // divide each element by the sum.
        for b in row.iter_mut() {
            *b /= sum
        }
    }
}

/// Softmax Gradient w.r.t. A
/// 
/// Performs the operation G *= B * (1. - B)
/// 
/// - B: Input 
/// - G: Gradient 
/// 
/// B _must_ be the output of the forward op.
pub fn softmax_wrt_a(
    b: &Array<f32, Ix2>,
    g: &mut Array<f32, Ix2>,
) {
    Zip::from(g).and(b).for_each(|g, b| {
        *g *= *b * (1. - *b)
    })
}
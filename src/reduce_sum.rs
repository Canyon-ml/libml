
use ndarray::{Array, Ix2, Dimension, Axis, RemoveAxis};

/// # ReduceSum Operator
/// 
/// ONNX definition: 
/// 
/// ## Summary
/// 
/// Computes the sum of the input tensor’s elements along 
/// the provided axis. The resulting tensor has the rank
/// of the input - 1. Input tensors of rank zero are valid. 
/// Reduction over an empty set of values yields 0.
/// 
/// The above behavior is similar to numpy, with the 
/// exception that numpy defaults keepdims to False instead of True.
/// 
/// Note: This operator only reduces by one dimension at a time.
/// 
/// ## Inputs
/// 
/// - data: Input 
/// - axes: Axis to reduce by
/// 
/// ## Outputs
/// 
/// - reduced: Output
/// 
/// ## Gradient
/// 
/// Because summation is linear, the gradient w.r.t.  
/// any element of data is 1. 
/// 
#[inline]
pub fn reduce_sum<D>(
    data: &Array<f32, D>,
    axis: Axis,

    reduced: &mut Array<f32, D::Smaller>,
) 
where
    D: RemoveAxis + Dimension,
{
    data.sum_axis(axis).clone_into(reduced);
}

// #[test]
// fn stuff() {
//     let a = vec![1., 2., 3., 4. ,5., 6.];
//     let b = Array::from_shape_vec((2, 3), a).unwrap();

//     let c = b.sum_axis(Axis(1));

//     panic!("{}", c);
// }
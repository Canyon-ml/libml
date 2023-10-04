
use ndarray::{Array, Dimension, RemoveAxis, Axis};

/// # ReduceMean Operator
/// 
/// ONNX definition: https://onnx.ai/onnx/operators/onnx__ReduceMean.html#reducemean
/// 
/// Computes the mean of the input tensors' elements along the specified Axis.
/// 
/// ## Inputs
/// 
/// - data: Input, Tensor to be reduced
/// - axis: the Axis to reduce by
/// 
/// ## Outputs
/// 
/// - reduced: The output tensor one dimension smaller than Data.
/// 
#[inline]
pub fn reduce_mean<D>(
    data: &Array<f32, D>,
    axis: Axis,

    reduced: &mut Array<f32, D::Smaller>,
)
where
    D: Dimension + RemoveAxis
{
    data.mean_axis(axis)
        .expect(&format!("The Mean of the Axis ({:?}) was Zero!", axis))
            .clone_into(reduced)
}

/// # ReduceMean Operator w.r.t. data
/// 
/// Computes the gradient by iterating by the removed axis and 
/// applying gy = *x * (1 / axis_len).
/// 
/// ## Inputs
/// 
/// - data: Input data of the forward op
/// - axis: Axis removed from data in the forward op
/// - gx: Input gradient
/// - gy: Output gradient
/// 
#[inline]
pub fn reduce_mean_wrt_data<D>(
    data: &Array<f32, D>,
    axis: Axis,

    gx: &Array<f32, D::Smaller>,
    gy: &mut Array<f32, D>,
)
where
    D: Dimension + RemoveAxis,
{
    let n = 1. / data.raw_dim()[axis.index()] as f32;

    for mut gy in gy.axis_iter_mut(axis) {
        for (x, y) in gx.iter().zip(gy.iter_mut()) {
            *y = *x * n;
        }
    }
}

// #[test]
// fn stuff() {
//     let a = vec![1., 2., 3., 4., 5., 6.];
//     let data = Array::from_shape_vec((2, 3), a.clone()).unwrap();
//     let mut reduced = Array::from_shape_vec(3, vec![0.0; 3]).unwrap();

//     super::reduce_mean(&data, Axis(0), &mut reduced);

//     let a = vec![9., 8., 7.];
//     let gx = Array::from_shape_vec(3, a).unwrap();
//     let mut gy = Array::from_shape_vec((2, 3), vec![0.0; 6]).unwrap();

//     reduce_mean_wrt_data(&data, Axis(0), &gx, &mut gy);

//     panic!("input: \n {} \n \n reduced: \n {} \n \n gradient: \n {}", data, reduced, gy);
// }
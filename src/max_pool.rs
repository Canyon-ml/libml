
use ndarray::{Array, Ix4, ArrayBase, Axis, Data, Dimension, Slice, s, Zip, azip};
use num_traits::Zero;

/// Max Pooling Operator
/// 
/// ONNX definition: https://onnx.ai/onnx/operators/onnx__MaxPool.html
pub fn max_pool(
    a: &Array<f32, Ix4>,
    b: &mut Array<f32, Ix4>,
    padx: usize,
    pady: usize,
    stridex: usize,
    stridey: usize, 
    filterx: usize,
    filtery: usize,
) {
    let padded = pad_with_zeros(a, vec![padx, pady, 0, 0]);

    todo!()
}

pub fn max_pool_wrt_a(

) {
    todo!()
}

/// Pad the edges of an array with zeros.
///
/// `pad_width` specifies the length of the padding at the beginning
/// and end of each axis.
///
/// **Panics** if `arr.ndim() != pad_width.len()`.
fn pad_with_zeros<A, S, D>(arr: &ArrayBase<S, D>, pad_width: Vec<usize>) -> Array<A, D>
where
    A: Clone + Zero,
    S: Data<Elem = A>,
    D: Dimension,
{
    assert_eq!(
        arr.ndim(),
        pad_width.len(),
        "Array ndim must match length of `pad_width`."
    );

    // Compute shape of final padded array.
    let mut padded_shape = arr.raw_dim();
    for (ax, (&ax_len, pad)) in arr.shape().iter().zip(&pad_width).enumerate() {
        padded_shape[ax] = ax_len + pad + pad;
    }

    let mut padded = Array::zeros(padded_shape);
    {
        // Select portion of padded array that needs to be copied from the
        // original array.
        let mut orig_portion = padded.view_mut();
        for (ax, pad) in pad_width.iter().enumerate() {
            // FIXME: This has a bug when `pad_hi` is 0. See @fzyzcjy's comment below.
            orig_portion
                .slice_axis_inplace(Axis(ax), Slice::from(*pad as isize..-(*pad as isize)));
        }
        // Copy the data from the original array.
        orig_portion.assign(arr);
    }
    padded
}

#[test]
fn stuff() {
    let a = Array::from_shape_vec((4, 4), vec![1., 2., 3., 4., 5., 6. ,7., 8., 9., 10., 11., 12., 13., 14., 15., 16.]).unwrap();

    let s = a.slice(s![2..4, 2..4]);

    panic!("{}, \n \n {}", a, s);
}

use ndarray::{Array4, Array2, Array3, Array1, Array, s, Axis};

pub fn conv(

) {

}

/// - A: CxHxW Tensor to be converted
/// - B: Output Tensor
pub(crate) fn im2col(
    a: &Array4<f32>,
    b: &mut Array3<f32>,

    kw: usize,
    kh: usize,

    xstride: usize,
    ystride: usize,
) {
    for (n, chw) in a.axis_iter_mut(Axis(0)).enumerate() {
        let mut cont = 0_usize;
        for i in 1..b.dim().0 + 1 {
            for j in 1..b.dim().1 + 1 {
                let patch = b.slice(s![
                    ..,
                    (i - 1) * ystride..((i - 1) * ystride + kh),
                    (j - 1) * xstride..((j - 1) * xstride + kw),
                ]);
                let patchrow_unwrap: Array1<f32> = Array::from_iter(patch.map(|a| *a));

                b.index_axis_mut(Axis(2), n).row_mut(cont).assign(&patchrow_unwrap);
                cont += 1;
            }
        }
    }
}

/// ## Inputs
/// 
/// - A: Input Data with shape (C, H, W)
pub(crate) fn col2im(
    a: &Array2<f32>,
    b: &mut Array3<f32>,
) {

}

// #[test]
// fn stuff() {
//     let a = vec![
//         1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
//         1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
//         1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
//         16., 15., 14., 13., 12., 11., 10., 9., 8., 7., 6., 5., 4., 3., 2., 1.,
//         16., 15., 14., 13., 12., 11., 10., 9., 8., 7., 6., 5., 4., 3., 2., 1.,
//         16., 15., 14., 13., 12., 11., 10., 9., 8., 7., 6., 5., 4., 3., 2., 1.,
//     ];

//     let a = Array::from_shape_vec((2, 3, 4, 4), a);
//     let b = Array::zeros((4 / 1 + 1, 4 / 1 + 1, ))
// }
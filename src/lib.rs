#![feature(portable_simd)]
#![feature(array_chunks)]

use rayon::prelude::*;
use std::simd::{f32x4, f32x8, simd_swizzle, u8x16, u8x8, usizex8, SimdFloat, StdFloat};

// Reference: https://gist.github.com/arifd/ea820ec97265a023e67a88b66955855d
pub fn yuv422_to_rgb24_chunk4(in_buf: &[u8], out_buf: &mut [u8]) {
    debug_assert!(out_buf.len() * 2 == in_buf.len() * 3);

    in_buf
        .par_chunks_exact(4) // FIXME: use par_array_chunks() when stabalized (https://github.com/rayon-rs/rayon/pull/789)
        .zip(out_buf.par_chunks_exact_mut(6))
        .for_each(|(ch, out)| {
            let y1 = ch[0];
            let y2 = ch[2];
            let u = ch[1];
            let v = ch[3];

            let [r, g, b] = yuv_to_rgb_simd(y1, u, v);
            out[0] = r;
            out[1] = g;
            out[2] = b;

            let [r, g, b] = yuv_to_rgb_simd(y2, u, v);
            out[3] = r;
            out[4] = g;
            out[5] = b;
        });
}

pub fn yuv422_to_rgb24_chunk16(in_buf: &[u8], out_buf: &mut [u8]) {
    assert!(out_buf.len() * 2 == in_buf.len() * 3);

    in_buf
        .par_chunks_exact(16) // FIXME: use par_array_chunks() when stabalized (https://github.com/rayon-rs/rayon/pull/789)
        .zip(out_buf.par_chunks_exact_mut(24))
        .for_each(|(yuv422, rgb24)| {
            let yuv422: [u8; 16] = yuv422.try_into().unwrap();
            yuv_to_rgb_chunk16(yuv422, rgb24);
        });

    // trailing case
    let remain_len = (in_buf.len() % 16) / 2;
    let in_buf_range = (in_buf.len() - remain_len * 2)..in_buf.len();
    let out_buf_range = (out_buf.len() - remain_len * 3)..out_buf.len();
    let in_trail_buf = &in_buf[in_buf_range];
    let out_trail_buf = &mut out_buf[out_buf_range];
    yuv422_to_rgb24_chunk4(in_trail_buf, out_trail_buf);
}

pub fn yuv_to_rgb_simd(y: u8, cb: u8, cr: u8) -> [u8; 3] {
    let yuv = f32x4::from_array([y as f32, cb as f32 - 128.0, cr as f32 - 128.0, 0.0]);

    // rec 709: https://mymusing.co/bt-709-yuv-to-rgb-conversion-color/
    let r = (yuv * f32x4::from_array([1.0, 0.0, 1.5748, 0.0])).reduce_sum();
    let g = (yuv * f32x4::from_array([1.0, -0.187324, -0.468124, 0.0])).reduce_sum();
    let b = (yuv * f32x4::from_array([1.0, 1.8556, 0.0, 0.0])).reduce_sum();

    let c0 = f32x4::splat(0.0);
    let c255 = f32x4::splat(255.0);
    let [r, g, b, _] = f32x4::from_array([r, g, b, 0.0])
        .simd_clamp(c0, c255)
        .round()
        .cast::<u8>()
        .to_array();
    [r, g, b]
}

pub fn yuv_to_rgb_chunk16(yuv422: [u8; 16], rgb24: &mut [u8]) {
    debug_assert!(rgb24.len() == 24);

    let vr = 1.5748;
    let ug = -0.187324;
    let vg = -0.468124;
    let ub = 1.8556;

    let yuv422 = u8x16::from_array(yuv422);
    let y_buf: u8x8 = simd_swizzle!(yuv422, [0, 2, 4, 6, 8, 10, 12, 14]);
    let u_buf: u8x8 = simd_swizzle!(yuv422, [1, 1, 5, 5, 9, 9, 13, 13]);
    let v_buf: u8x8 = simd_swizzle!(yuv422, [3, 3, 7, 7, 11, 11, 15, 15]);

    let y_buf: f32x8 = y_buf.cast();
    let u_buf: f32x8 = u_buf.cast() - f32x8::splat(128.0);
    let v_buf: f32x8 = v_buf.cast() - f32x8::splat(128.0);

    let r_buf = y_buf + v_buf * f32x8::splat(vr);
    let g_buf = y_buf + u_buf * f32x8::splat(ug) + v_buf * f32x8::splat(vg);
    let b_buf = y_buf + u_buf * f32x8::splat(ub);

    let c0 = f32x8::splat(0.0);
    let c255 = f32x8::splat(255.0);
    let r_buf = r_buf.simd_clamp(c0, c255);
    let g_buf = g_buf.simd_clamp(c0, c255);
    let b_buf = b_buf.simd_clamp(c0, c255);

    let r_buf: u8x8 = r_buf.cast();
    let g_buf: u8x8 = g_buf.cast();
    let b_buf: u8x8 = b_buf.cast();

    r_buf.scatter(rgb24, usizex8::from_array([0, 3, 6, 9, 12, 15, 18, 21]));
    g_buf.scatter(rgb24, usizex8::from_array([1, 4, 7, 10, 13, 16, 19, 22]));
    b_buf.scatter(rgb24, usizex8::from_array([2, 5, 8, 11, 14, 17, 20, 23]));
}

// pub fn yuv422_to_rgb24(in_buf: &[u8], out_buf: &mut [u8]) {
//     assert!(out_buf.len() * 2 == in_buf.len() * 3);

//     in_buf
//         .par_chunks_exact(4) // FIXME: use par_array_chunks() when stabalized (https://github.com/rayon-rs/rayon/pull/789)
//         .zip(out_buf.par_chunks_exact_mut(6))
//         .for_each(|(ch, out)| {
//             let y1 = ch[0];
//             let y2 = ch[2];
//             let u = ch[1];
//             let v = ch[3];

//             let [r, g, b] = yuv_to_rgb(y1, u, v);
//             out[0] = r;
//             out[1] = g;
//             out[2] = b;

//             let [r, g, b] = yuv_to_rgb(y2, u, v);
//             out[3] = r;
//             out[4] = g;
//             out[5] = b;
//         });
// }

// fn yuv_to_rgb(y: u8, u: u8, v: u8) -> [u8; 3] {
//     use yuv::{
//         color::{MatrixCoefficients, Range},
//         convert::RGBConvert,
//         RGB, YUV,
//     };

//     let yuv = YUV { y, u, v };
//     let converter = RGBConvert::<u8>::new(Range::Limited, MatrixCoefficients::BT709).unwrap();
//     let RGB { r, g, b } = converter.to_rgb(yuv);
//     [r, g, b]
// }

// fn clamp(val: f32) -> f32 {
//     val.clamp(0.0, 255.0)
// }

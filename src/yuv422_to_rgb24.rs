use rayon::prelude::*;
use std::simd::{
    f32x2, f32x8, simd_swizzle, u8x16, u8x4, u8x8, usizex8, Simd, SimdFloat, SimdUint,
};

#[allow(non_camel_case_types)]
type u8x2 = Simd<u8, 2>;

#[allow(non_camel_case_types)]
type usizex2 = Simd<usize, 2>;

// Reference: https://gist.github.com/arifd/ea820ec97265a023e67a88b66955855d
pub fn yuyv422_to_rgb24_chunk4_many(in_buf: &[u8], out_buf: &mut [u8]) {
    debug_assert!(in_buf.len() % 4 == 0);
    debug_assert!(out_buf.len() * 2 == in_buf.len() * 3);

    in_buf
        .par_chunks_exact(4) // FIXME: use par_array_chunks() when stabalized (https://github.com/rayon-rs/rayon/pull/789)
        .zip(out_buf.par_chunks_exact_mut(6))
        .for_each(|(in_buf, out_buf)| {
            let in_buf: [u8; 4] = in_buf.try_into().unwrap();
            yuyv422_to_rgb24_chunk4_single(in_buf, out_buf);
        });
}

pub fn yuyv422_to_rgb24_chunk16_many(in_buf: &[u8], out_buf: &mut [u8]) {
    assert!(out_buf.len() * 2 == in_buf.len() * 3);

    in_buf
        .par_chunks_exact(16) // FIXME: use par_array_chunks() when stabalized (https://github.com/rayon-rs/rayon/pull/789)
        .zip(out_buf.par_chunks_exact_mut(24))
        .for_each(|(yuv422, rgb24)| {
            let yuv422: [u8; 16] = yuv422.try_into().unwrap();
            yuyv422_to_rgb24_chunk16_single(yuv422, rgb24);
        });

    // trailing case
    let remain_len = (in_buf.len() % 16) / 2;
    let in_buf_range = (in_buf.len() - remain_len * 2)..in_buf.len();
    let out_buf_range = (out_buf.len() - remain_len * 3)..out_buf.len();
    let in_trail_buf = &in_buf[in_buf_range];
    let out_trail_buf = &mut out_buf[out_buf_range];
    yuyv422_to_rgb24_chunk4_many(in_trail_buf, out_trail_buf);
}

pub fn yuyv422_to_rgb24_chunk4_single(yuv422: [u8; 4], rgb24: &mut [u8]) {
    debug_assert!(rgb24.len() == 6);

    let vr = 1.5748;
    let ug = -0.187324;
    let vg = -0.468124;
    let ub = 1.8556;

    let yuv422 = u8x4::from_array(yuv422);
    let y_buf: u8x2 = simd_swizzle!(yuv422, [0, 2]);
    let u_buf: u8x2 = simd_swizzle!(yuv422, [1, 1]);
    let v_buf: u8x2 = simd_swizzle!(yuv422, [3, 3]);

    let y_buf: f32x2 = y_buf.cast();
    let u_buf: f32x2 = u_buf.cast() - f32x2::splat(128.0);
    let v_buf: f32x2 = v_buf.cast() - f32x2::splat(128.0);

    let r_buf = y_buf + v_buf * f32x2::splat(vr);
    let g_buf = y_buf + u_buf * f32x2::splat(ug) + v_buf * f32x2::splat(vg);
    let b_buf = y_buf + u_buf * f32x2::splat(ub);

    let c0 = f32x2::splat(0.0);
    let c255 = f32x2::splat(255.0);
    let r_buf = r_buf.simd_clamp(c0, c255);
    let g_buf = g_buf.simd_clamp(c0, c255);
    let b_buf = b_buf.simd_clamp(c0, c255);

    let r_buf: u8x2 = r_buf.cast();
    let g_buf: u8x2 = g_buf.cast();
    let b_buf: u8x2 = b_buf.cast();

    r_buf.scatter(rgb24, usizex2::from_array([0, 3]));
    g_buf.scatter(rgb24, usizex2::from_array([1, 4]));
    b_buf.scatter(rgb24, usizex2::from_array([2, 5]));
}

pub fn yuyv422_to_rgb24_chunk16_single(yuv422: [u8; 16], rgb24: &mut [u8]) {
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

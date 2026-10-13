// Single-precision libm wrappers matching the Rust reference bit-for-bit.
// Rust's f32::{sin,cos,log2,ln,exp,log10,tanh,pow,sqrt} lower to the platform
// C library's float routines (sinf/cosf/... on glibc); Go's math package
// computes in float64 and rounds, which differs from sinf/cosf on ~1% of
// arguments by 1 ulp — enough to flip rare quantizer ties in the encoder.
// The cgo build therefore calls the C float routines directly; the pure-Go
// (!cgo) build falls back to the float64-backed wrappers and stays within
// tight float tolerance of the reference, but its ENCODER bitstream is not
// guaranteed byte-exact (the decoder is unaffected: it never calls these).
package mlow

/*
#include <math.h>
#cgo LDFLAGS: -lm
*/
import "C"

func cos32(x float32) float32  { return float32(C.cosf(C.float(x))) }
func sin32(x float32) float32  { return float32(C.sinf(C.float(x))) }
func f32Sin(x float32) float32 { return sin32(x) }
func f32Cos(x float32) float32 { return cos32(x) }

func f32Log2(x float32) float32 { return float32(C.log2f(C.float(x))) }
func f32Ln(x float32) float32   { return float32(C.logf(C.float(x))) }
func f32Exp(x float32) float32  { return float32(C.expf(C.float(x))) }
func f32Tanh(x float32) float32 { return float32(C.tanhf(C.float(x))) }
func f32Pow(x, y float32) float32 {
	return float32(C.powf(C.float(x), C.float(y)))
}
func f32Sqrt(x float32) float32 { return float32(C.sqrtf(C.float(x))) }
func f32Log10(x float32) float32 {
	return float32(C.log10f(C.float(x)))
}

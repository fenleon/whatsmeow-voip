//go:build !cgo

// Pure-Go fallbacks for the f32 libm wrappers (see f32libm.go): float64 math
// rounded to float32. Same tight-tolerance envelope as the rest of the port;
// the encoder bitstream is not bit-exact to the Rust reference in this build.
package mlow

import "math"

func cos32(x float32) float32  { return float32(math.Cos(float64(x))) }
func sin32(x float32) float32  { return float32(math.Sin(float64(x))) }
func f32Sin(x float32) float32 { return sin32(x) }
func f32Cos(x float32) float32 { return cos32(x) }

func f32Log2(x float32) float32 { return float32(math.Log2(float64(x))) }
func f32Ln(x float32) float32   { return float32(math.Log(float64(x))) }
func f32Exp(x float32) float32  { return float32(math.Exp(float64(x))) }
func f32Tanh(x float32) float32 { return float32(math.Tanh(float64(x))) }
func f32Pow(x, y float32) float32 {
	return float32(math.Pow(float64(x), float64(y)))
}
func f32Sqrt(x float32) float32 { return float32(math.Sqrt(float64(x))) }
func f32Log10(x float32) float32 {
	return float32(math.Log10(float64(x)))
}

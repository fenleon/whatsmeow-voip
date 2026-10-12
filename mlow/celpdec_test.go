package mlow

import (
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"math"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func f32Slice(v []float64) []float32 {
	out := make([]float32, len(v))
	for i, x := range v {
		out[i] = float32(x)
	}
	return out
}

type gennoiseVector struct {
	Voiced   int       `json:"voiced"`
	ExcPre   []float64 `json:"exc_pre"`
	LSF      []float64 `json:"lsf"`
	Noise    []float64 `json:"noise"`
	Nrgres   float64   `json:"nrgres"`
	FcbgIdx  int32     `json:"fcbg_idx"`
	SfPulses int32     `json:"sf_pulses"`
	NormBr   float64   `json:"norm_br"`
	SeedOut  int32     `json:"seed_out"`
	NgIn     ngState   `json:"ng_in"`
	NgOut    ngState   `json:"ng_out"`
}

type ngState struct {
	EnvSmth       float64   `json:"env_smth"`
	EnvLast       float64   `json:"env_last"`
	OutStateUV    []float64 `json:"out_state_uv"`
	OutStateV     []float64 `json:"out_state_v"`
	CorrSmth      []float64 `json:"corr_smth"`
	ShapeState    []float64 `json:"shape_state"`
	PrevVoiced    int       `json:"prev_voiced"`
	SinceUnvoiced int32     `json:"since_unvoiced"`
	RandSeed      int32     `json:"rand_seed"`
}

func ngFrom(v ngState) *NoiseGenerator {
	return &NoiseGenerator{
		EnvSmth:       float32(v.EnvSmth),
		EnvLast:       float32(v.EnvLast),
		OutStateUV:    [2]float32(f32Slice2(v.OutStateUV)),
		OutStateV:     [2]float32(f32Slice2(v.OutStateV)),
		CorrSmth:      [3]float32(f32Slice3(v.CorrSmth)),
		ShapeState:    [2]float32(f32Slice2(v.ShapeState)),
		PrevVoiced:    v.PrevVoiced != 0,
		SinceUnvoiced: v.SinceUnvoiced,
		RandSeed:      v.RandSeed,
	}
}

func f32Slice2(v []float64) []float32 {
	s := f32Slice(v)
	return []float32{s[0], s[1]}
}

func f32Slice3(v []float64) []float32 {
	s := f32Slice(v)
	return []float32{s[0], s[1], s[2]}
}

// TestGenNoiseMatchesVectors: gen_noise is PRNG- and state-driven, so the
// only deterministic validation is bit-exactness against the reference
// vectors (noise, seed transition, output state), across voiced and unvoiced
// paths.
func TestGenNoiseMatchesVectors(t *testing.T) {
	data, err := os.ReadFile("../vectors/mlow/gennoise_vectors.json")
	require.NoError(t, err)
	var vectors []gennoiseVector
	require.NoError(t, json.Unmarshal(data, &vectors))
	require.NotEmpty(t, vectors)

	// fcbgains_uv[ix] = 10^(0.05*(ix-90)), ix in 0..=90.
	fcbgainsUV := make([]float32, 91)
	for ix := range fcbgainsUV {
		fcbgainsUV[ix] = float32(math.Pow(10, float64(0.05*(float32(ix)-90.0))))
	}

	checked := struct{ v, uv0, uvp int }{}
	for _, rec := range vectors {
		voiced := rec.Voiced == 1
		exc := f32Slice(rec.ExcPre)
		lsf := f32Slice(rec.LSF)
		noiseExpected := f32Slice(rec.Noise)
		ng := ngFrom(rec.NgIn)
		ngOutExpected := ngFrom(rec.NgOut)
		var noise [smplMaxSFLen]float32
		CelpGenNoise(ng, exc, 80, voiced, rec.SfPulses,
			float32(rec.Nrgres), rec.FcbgIdx, lsf, float32(rec.NormBr),
			fcbgainsUV, noise[:])

		// Seed transition is bit-exact (proves the PRNG + gen_rand_pulses
		// call count).
		assert.Equal(t, rec.SeedOut, ng.RandSeed, "seed_out (voiced=%v np=%d)", voiced, rec.SfPulses)
		assert.Equal(t, ngOutExpected.RandSeed, ng.RandSeed, "ng_out.rand_seed")

		// noise[80] bit-exact (tiny tolerance for f32 transcendental
		// last-ULP across libm).
		for i := 0; i < 80; i++ {
			assert.Less(t, math.Abs(float64(noise[i]-noiseExpected[i])), 1e-6,
				"noise[%d] mismatch (voiced=%v np=%d): got=%v ref=%v", i, voiced, rec.SfPulses, noise[i], noiseExpected[i])
		}
		assert.Less(t, math.Abs(float64(ng.EnvLast-ngOutExpected.EnvLast)), 1e-6,
			"ng_out.env_last (voiced=%v np=%d)", voiced, rec.SfPulses)
		for k := 0; k < 2; k++ {
			assert.Less(t, math.Abs(float64(ng.OutStateUV[k]-ngOutExpected.OutStateUV[k])), 1e-6,
				"ng_out.out_state_uv[%d]", k)
			assert.Less(t, math.Abs(float64(ng.OutStateV[k]-ngOutExpected.OutStateV[k])), 1e-6,
				"ng_out.out_state_v[%d]", k)
		}

		switch {
		case voiced:
			checked.v++
		case rec.SfPulses == 0:
			checked.uv0++
		default:
			checked.uvp++
		}
	}
	assert.Greater(t, checked.v, 0, "vectors must exercise the voiced path")
	assert.Greater(t, checked.uv0, 0, "vectors must exercise the np=0 unvoiced path")
	assert.Greater(t, checked.uvp, 0, "vectors must exercise the np>0 unvoiced path")
}

func rf32(b []byte, o *int) float32 {
	v := math.Float32frombits(binary.LittleEndian.Uint32(b[*o:]))
	*o += 4
	return v
}

func ri32(b []byte, o *int) int32 {
	v := int32(binary.LittleEndian.Uint32(b[*o:]))
	*o += 4
	return v
}

// TestHarmcombPostfilterMatchesVectors validates the HP postfilter against
// the instrumented reference decoder dumps. The reference is built with
// -ffast-math -mavx (reassociated recursive accumulations), so we assert the
// error stays well under the i16 output quantization step.
func TestHarmcombPostfilterMatchesVectors(t *testing.T) {
	var i16LSB float32 = 1.0 / 32768.0
	data, err := os.ReadFile("../vectors/mlow/hp_postfilter_vectors.raw")
	require.NoError(t, err)
	o := 0
	count := ri32(data, &o)
	require.Greater(t, count, int32(0))
	worst := float32(0.0)
	for i := int32(0); i < count; i++ {
		ri32(data, &o) // packet
		ri32(data, &o) // frame
		var lags [8]float32
		for j := range lags {
			lags[j] = rf32(data, &o)
		}
		lo1 := rf32(data, &o)
		lo2 := rf32(data, &o)
		var hp [4]float32
		for j := range hp {
			hp[j] = rf32(data, &o)
		}
		lagOld := rf32(data, &o)
		var xOld [hpFrameLen]float32
		for j := range xOld {
			xOld[j] = rf32(data, &o)
		}
		var coefMA, coefAR [3]float32
		for j := 0; j < 3; j++ {
			coefMA[j] = rf32(data, &o)
		}
		for j := 0; j < 3; j++ {
			coefAR[j] = rf32(data, &o)
		}
		yPre := make([]float32, hpFrameLen)
		for j := range yPre {
			yPre[j] = rf32(data, &o)
		}
		yPost := make([]float32, hpFrameLen)
		for j := range yPost {
			yPost[j] = rf32(data, &o)
		}

		lag := float32(0.0)
		if lags[0] > 0.0 {
			var sl, sll float32
			for _, l := range lags {
				sl += l
				sll += l * l
			}
			lag = sll / sl
		}

		st := &HpPostfilterState{
			stateLoEmph1: lo1,
			stateLoEmph2: lo2,
			stateHP:      hp,
			lagOld:       lagOld,
			xOld:         xOld,
			coefMA:       coefMA,
			coefAR:       coefAR,
		}
		out := make([]float32, hpFrameLen)
		SmplHarmcombPostfilter(st, yPre, hpFrameLen, lag, out)
		for j := range out {
			diff := out[j] - yPost[j]
			if diff < 0 {
				diff = -diff
			}
			if diff > worst {
				worst = diff
			}
		}
	}
	assert.Less(t, worst, i16LSB,
		"hp_postfilter diverges from reference by %e (>= i16 LSB %e)", worst, i16LSB)
}

// TestExcPreMatchesVectors validates the pre-noise excitation (FCB pulses *
// gain + voiced ACB) against the reference exc_pre dump, per subframe. This
// proves the excitation domain and the voiced ACB/LTP synthesis are faithful,
// independent of the PRNG-driven noise.
func TestExcPreMatchesVectors(t *testing.T) {
	data, err := os.ReadFile("../vectors/mlow/exc_pre_lags.json")
	require.NoError(t, err)
	var captures []excPreCapture
	require.NoError(t, json.Unmarshal(data, &captures))

	framesData, err := os.ReadFile("../vectors/mlow/inbound_capture_frames.json")
	require.NoError(t, err)
	var frames []string
	require.NoError(t, json.Unmarshal(framesData, &frames))

	built := builtTables(t)
	cc := ccBuilt(t)
	mem := memBuilt(t)
	synth := &built.Synth

	type key struct{ packet, frame, sf int64 }
	cmap := make(map[key]*excPreCapture, len(captures))
	for i := range captures {
		c := &captures[i]
		cmap[key{int64(c.Packet), int64(c.Frame), int64(c.Sf)}] = c
	}

	var lstate SmplLsfState
	celp := NewCelpDecState()
	var prevNLSF []float32

	counts := struct{ uvOK, uvBad, vOK, vBad int }{}
	worst := float32(0)
	for packet, hexFrame := range frames {
		frame, err := hex.DecodeString(hexFrame)
		require.NoError(t, err)
		if len(frame) == 0 {
			continue
		}
		toc := ParseMlowToc(frame[0])
		if toc.StdOpus || toc.SID || !toc.Active {
			continue
		}
		config := int(frame[0]>>2) & 1
		lowRate := frame[0]>>2&1 != 0
		dec := NewRangeDecoder(frame[1:])
		for f := 0; f < 3; f++ {
			lsf := DecodeSmplLsf(dec, &built.Tables, &lstate, config, f)
			pulses := DecodeSmplPulses(dec, cc, 320, 4, 1, int32(config), lsf.Stage1)
			voiced := lsf.Stage1 == 1
			params := CelpDecParams{
				Voiced:      voiced,
				SfPulses:    pulses.Subfr,
				TotalPulses: pulses.Subfr[0] + pulses.Subfr[1] + pulses.Subfr[2] + pulses.Subfr[3],
			}
			if voiced {
				pr := DecodeSmplPitch(dec, mem, cc, &lstate, 320, 4, int32(config), pulses.Subfr)
				for b := 0; b < 8; b++ {
					v := float64(pr.BlockLags[b])*0.5 + 32.0
					if v > 320.0 {
						v = 320.0
					}
					params.BlockLags[b] = float32(v)
				}
				for sf := 0; sf < 4; sf++ {
					params.AcbgIdx[sf] = pr.GainIdx[sf]
					if pr.FiltIdx[sf] > 0 {
						params.FcbgIdx[sf] = pr.FiltIdx[sf]
					}
				}
			} else {
				g := DecodeSmplGains(dec, cc, 4, pulses.Subfr)
				params.NrgresDbqQ14 = g.GainQ
				params.FcbgIdx = g.NrgRes
			}
			nlsf := ReconstructNLSF(synth, int(lsf.Stage1), config, int(lsf.Grid), &lsf.Stage2, prevNLSF)
			celp.DbgExcPre = celp.DbgExcPre[:0]
			var sig [smplIntfLen]float32
			celp.SynthFrame(nlsf, int(lsf.Extra), pulses.Pulses, &params, lowRate, 320, sig[:],
				NewHpPostfilterState())
			prevNLSF = nlsf
			for sf := 0; sf < 4; sf++ {
				c, ok := cmap[key{int64(packet), int64(f), int64(sf)}]
				if !ok {
					continue
				}
				if voiced {
					c0 := float32(c.Lags[0])
					c1 := float32(c.Lags[1])
					assert.Equal(t, [2]float32{params.BlockLags[2*sf], params.BlockLags[2*sf+1]},
						[2]float32{c0, c1},
						"per-block lags diverge at pkt=%d f=%d sf=%d", packet, f, sf)
				}
				var cexc [smplSubfrLen]float32
				for _, pair := range c.Nz {
					cexc[int(pair[0])] = float32(pair[1])
				}
				base := sf * smplSubfrLen
				bad := false
				for i := 0; i < smplSubfrLen; i++ {
					d := celp.DbgExcPre[base+i] - cexc[i]
					if d < 0 {
						d = -d
					}
					if d > worst {
						worst = d
					}
					if d > 2e-5 {
						bad = true
					}
				}
				switch {
				case voiced && bad:
					counts.vBad++
				case voiced:
					counts.vOK++
				case bad:
					counts.uvBad++
				default:
					counts.uvOK++
				}
			}
		}
	}
	assert.Equal(t, 0, counts.uvBad,
		"unvoiced exc_pre diverges from reference (%d subframes)", counts.uvBad)
	assert.Equal(t, 0, counts.vBad,
		"voiced exc_pre diverges from reference (%d subframes)", counts.vBad)
}

type excPreCapture struct {
	Packet int          `json:"packet"`
	Frame  int          `json:"frame"`
	Sf     int          `json:"sf"`
	Lags   []float64    `json:"lags"`
	Nz     [][2]float64 `json:"nz"`
}

// Self-contained portable mixed-radix complex FFT (no external deps), ported
// from whatsapp-rust (MIT) wacore/src/voip/mlow/smpl_perc.rs:312-600; MPL-2.0
// for this port. Shared scratch for the 512-pt LPC front-end (smpl_lpc) and
// the 576-pt perceptual model (smpl_perc). The reference C used PFFFT; this
// portable FFT differs only in internal rounding, so downstream outputs match
// the C to a tight float tolerance, not bit-for-bit.
package mlow

type cpx struct{ re, im float32 }

func cpxAdd(a, b cpx) cpx { return cpx{a.re + b.re, a.im + b.im} }

func cpxMul(a, b cpx) cpx {
	return cpx{a.re*b.re - a.im*b.im, a.re*b.im + a.im*b.re}
}

// smallestFactor returns the smallest prime factor of n (>= 2). For our use n
// is always a composite of 2 and 3.
func smallestFactor(n int) int {
	if n%2 == 0 {
		return 2
	}
	for p := 3; p*p <= n; p += 2 {
		if n%p == 0 {
			return p
		}
	}
	return n
}

// fftTwiddles holds precomputed twiddle factors for one (top-N, sign) FFT.
// Each table reproduces the exact inline angle arithmetic (same f32 op order),
// so reading from it is bit-identical to recomputing inline.
type fftTwiddles struct {
	// combine: per visited length n, the combine-step twiddles indexed
	// k*p+q (length n*p).
	combine []fftTwiddleTable
	// base: per prime base length n, the base-DFT twiddles indexed
	// k*n+j (length n*n).
	base []fftTwiddleTable
}

type fftTwiddleTable struct {
	n   int
	tab []cpx
}

// combineTwiddle computes the combine twiddle for (n, k, q, sign) exactly as
// the inline butterfly does.
func combineTwiddle(n, k, q int, sign float32) cpx {
	ang := sign * 2.0 * smplPi * float32(k) * float32(q) / float32(n)
	return cpx{cos32(ang), sin32(ang)}
}

// baseTwiddle computes the prime-base twiddle for (n, k, j, sign) exactly as
// the inline base DFT does (two-step: ang_k = sign*2π*k/n, then ang = ang_k*j).
func baseTwiddle(n, k, j int, sign float32) cpx {
	angK := sign * 2.0 * smplPi * float32(k) / float32(n)
	ang := angK * float32(j)
	return cpx{cos32(ang), sin32(ang)}
}

// newFftTwiddles builds the twiddle tables for the chain of n values the
// recursion visits from top down, for the given sign.
func newFftTwiddles(top int, sign float32) *fftTwiddles {
	tw := &fftTwiddles{}
	for n := top; n > 1; {
		p := smallestFactor(n)
		if p == n {
			tab := make([]cpx, n*n)
			for k := 0; k < n; k++ {
				for j := 0; j < n; j++ {
					tab[k*n+j] = baseTwiddle(n, k, j, sign)
				}
			}
			tw.base = append(tw.base, fftTwiddleTable{n, tab})
			break
		}
		tab := make([]cpx, n*p)
		for k := 0; k < n; k++ {
			for q := 0; q < p; q++ {
				tab[k*p+q] = combineTwiddle(n, k, q, sign)
			}
		}
		tw.combine = append(tw.combine, fftTwiddleTable{n, tab})
		n /= p
	}
	return tw
}

func (tw *fftTwiddles) combineFor(n int) []cpx {
	for _, t := range tw.combine {
		if t.n == n {
			return t.tab
		}
	}
	panic("combine twiddle table missing for n")
}

func (tw *fftTwiddles) baseFor(n int) []cpx {
	for _, t := range tw.base {
		if t.n == n {
			return t.tab
		}
	}
	panic("base twiddle table missing for n")
}

// fftScratch is the reusable FFT workspace (arena backs fftRec's per-level
// sub buffers; cin/spec/tout back the real-FFT pack/unpack; twFwd/twBwd are
// the precomputed twiddles for the two signs).
type fftScratch struct {
	arena, cin, spec, tout []cpx
	twFwd, twBwd           *fftTwiddles
}

func newFftScratch(n int) *fftScratch {
	return &fftScratch{
		arena: make([]cpx, fftArenaLen(n)),
		cin:   make([]cpx, n),
		spec:  make([]cpx, n),
		tout:  make([]cpx, n),
		twFwd: newFftTwiddles(n, -1.0),
		twBwd: newFftTwiddles(n, 1.0),
	}
}

// fftArenaLen is the worst-case fftRec arena length: at each level we carve n
// cpx for sub then recurse on m = n/p (children are sequential, so they reuse
// the remainder; the bound is additive down one path, not across siblings).
func fftArenaLen(n int) int {
	if n <= 1 {
		return 0
	}
	p := smallestFactor(n)
	if p == n {
		return 0 // prime base case allocates nothing
	}
	return n + fftArenaLen(n/p)
}

// fftRec is the recursive mixed-radix Cooley-Tukey DFT. x[i] are the n input
// samples at stride `stride` within the parent array; out is contiguous.
// scratch is the remaining arena: each non-base level splits off the first n
// entries for its sub buffer and passes the rest down (siblings are
// sequential, so the rest is safely shared).
func fftRec(x []cpx, stride, n int, out []cpx, scratch []cpx, tw *fftTwiddles) {
	if n == 1 {
		out[0] = x[0]
		return
	}
	p := smallestFactor(n)
	if p == n {
		// Prime (or n with no small factor): naive O(n^2) DFT base case.
		bt := tw.baseFor(n)
		for k := 0; k < n; k++ {
			acc := cpx{}
			row := k * n
			for j := 0; j < n; j++ {
				acc = cpxAdd(acc, cpxMul(x[j*stride], bt[row+j]))
			}
			out[k] = acc
		}
		return
	}
	m := n / p
	sub, rest := scratch[:n], scratch[n:]
	// Compute p sub-DFTs of length m over the decimated inputs.
	for q := 0; q < p; q++ {
		fftRec(x[q*stride:], stride*p, m, sub[q*m:(q+1)*m], rest, tw)
	}
	// Combine: out[k] = sum_q twiddle(q,k) * sub_q[k mod m].
	ct := tw.combineFor(n)
	for k := 0; k < n; k++ {
		kmod := k % m
		acc := cpx{}
		row := k * p
		for q := 0; q < p; q++ {
			acc = cpxAdd(acc, cpxMul(sub[q*m+kmod], ct[row+q]))
		}
		out[k] = acc
	}
}

// cfft computes the complex FFT of a mixed-radix length into out.
// sign=-1 forward, +1 inverse.
func cfft(input, out []cpx, sign float32, arena []cpx, tw *fftTwiddles) {
	fftRec(input, 1, len(input), out, arena, tw)
}

// rfftForwardOrderedSC is the forward real FFT of n real samples, re-packed
// into the ordered real layout: f[0]=DC.re, f[1]=Nyquist.re, then [re,im]
// pairs for bins 1..n/2-1. Output length is n.
func rfftForwardOrderedSC(time []float32, f []float32, sc *fftScratch) {
	n := len(time)
	for i := 0; i < n; i++ {
		sc.cin[i] = cpx{time[i], 0}
	}
	cfft(sc.cin, sc.spec, -1.0, sc.arena, sc.twFwd)
	f[0] = sc.spec[0].re
	f[1] = sc.spec[n/2].re
	for i := 1; i < n/2; i++ {
		f[2*i] = sc.spec[i].re
		f[2*i+1] = sc.spec[i].im
	}
}

// rfftBackwardOrderedSC is the inverse real FFT consuming the ordered real
// layout (as produced above) and producing n real time samples, unnormalized
// (BACKWARD(FORWARD(x)) = n*x).
func rfftBackwardOrderedSC(f []float32, time []float32, sc *fftScratch) {
	n := len(f)
	sc.spec[0] = cpx{f[0], 0}
	sc.spec[n/2] = cpx{f[1], 0}
	for i := 1; i < n/2; i++ {
		re, im := f[2*i], f[2*i+1]
		sc.spec[i] = cpx{re, im}
		sc.spec[n-i] = cpx{re, -im} // conjugate symmetry
	}
	cfft(sc.spec, sc.tout, 1.0, sc.arena, sc.twBwd)
	for i := 0; i < n; i++ {
		time[i] = sc.tout[i].re
	}
}

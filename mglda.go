package mglda

import (
	"math/rand"
	"sync"

	"github.com/skelterjohn/go.matrix"
)

const (
	globalTopic = "gl"
	localTopic  = "loc"
)

var (
	wg = sync.WaitGroup{}
)

type Document struct {
	sentenses []Sentense
}

type Sentense struct {
	words []int
}

type MGLDA struct {
	GlobalK        int
	LocalK         int
	Gamma          float64
	GlobalAlpha    float64
	LocalAlpha     float64
	GlobalAlphaMix float64
	LocalAlphaMix  folat64
	GlobalBeta     float64
	LocalBeta      float64
	Docs           []*Document
	T              float64
	W              float64
	Inflation      float64

	Vdsn [][][]int // sliding window for each word
	Rdsn [][][]string
	Zdsn [][][]int

	Nglzw *matrix.DenseMatrix
	Nglz  *matrix.DenseMatrix

	Ndsv  [][][]float64
	Nds   [][]float64
	Ndvgl *matrix.DenseMatrix
	Ndv   *matrix.DenseMatrix

	Ndglz *matrix.DenseMatrix
	Ndgl  *matrix.DenseMatrix

	Nloczw  *matrix.DenseMatrix
	Nlocz   *matrix.DenseMatrix
	Ndvloc  *matrix.DenseMatrix
	Ndvlocz [][][]float64
}

// Inference runs a go routine for each doc.
func (m *MGLDA) Inference() {
	for d, doc := range m.Docs {
		wg.Add(1)
		go func(d int, d *Document) {
			defer wg.Done()

			for s, sent := range doc {
				for w, wd := range sent {
					v := m.Vdsn[d][s][w]
					r := m.Rdsn[d][s][w]
					z := m.Zdsn[d][s][w]

					if r == globalTopic {
						m.Nglzw.Set(z, wd, m.Nglzw.Get(z, wd)-1)
						m.Nglz.Set(z, m.Nglz.Get(z)-1)
						m.Ndvgl.Set(d, s-v, m.Ndvgl.Get(d, s-v)-1)
						m.Ndglz.Set(d, z, m.Ndglz.Get(d, z)-1)
						m.Ndgl.Set(d, m.Ndgl.Get(d)-1)
					} else {
						m.Nloczw.Set(z, word, m.Nloczw.Get(z, word)-1)
						m.Nlocz.Set(z, m.Nlocz.Get(z)-1)
						m.Ndvloc.Set(d, s-v, m.Ndvloc.Get(d, s-v)-1)
						m.Ndvlocz[d][s-v][z] -= 1
					}
					m.Ndsv[d][s][v] -= 1
					m.Nds[d][s] -= 1
					m.Ndv[d][s-v] -= 1

					pvrz := []float64{}
					newVs := []int{}
					newRs := []string{}
					newZs := []int{}
					for vt := 0; vt < m.T; vt++ {
						newV = vt
						for zt := 0; zt < m.GlobalK; zt++ {
							newVs = append(newVs, vt)
							newRs = append(newRs, globalTopic)
							newZs = append(newZs, zt)

							term1 := (m.Nglzw.Get(zt, wd) + m.GlobalBeta) / (m.Nglz.Get(zt) + m.W*m.GlobalBeta)
							term2 := (m.Ndsv[d][s][vt] + m.Gamma) / (m.Nds[d][s] + m.T*m.Gamma)
							term3 := (m.Ndvgl.Get(d, s+vt) + m.GlobalAlpha) / (m.Ndv.Get(d, s+vt) + m.GlobalAlphaMix + m.LocalAlphaMix)
							term4 := (m.Ndglz.Get(d, zt) + m.GlobalAlpha) / (m.Ndgl.Get(d) + m.GlobalK*m.GlobalAlpha)
							pvrz = append(pvrz, term1*term2*term3*term4)
						}
						for zt := 0; zt < m.LocalK; zt++ {
							newVs = append(newVs, vt)
							newRs = append(newRs, localTopic)
							newZs = append(newZs, zt)

							term1 := (m.Nloczw.Get(zt, wd) + m.LocalBeta) / (m.Nlocz.Get(zt) + m.W*m.LocalBeta)
							term2 := (m.Ndsv[d][s][vt] + m.Gamma) / (m.Nds[d][s] + m.T*m.Gamma)
							term3 := (m.Ndvloc.Get(d, s+vt) + m.LocalAlphaMix) / (m.Ndv.Get(d, s+vt) + m.GlobalAlphaMix + m.LocalAlphaMix)
							term4 := (m.Ndvlocz[d][s+vt][zt] + m.LocalAlpha) / (m.Ndvloc.Get(d, s+vt) + m.LocalK*m.LocalAlpha)
							pvrz = append(pvrz, term1*term2*term3*term4)
						}
					}

					randIdx := rand.Intn(len(newZs))
					newV := newVs[randIdx]
					newR := newRs[randIdx]
					newZ := newZs[randIdx]

					// update
					if newR == globalTopic {
						m.Nglzw.Set(newZ, wd, m.Nglzw.Get(newZ, wd)+1)
						m.Nglz.Set(newZ, m.Nglz.Get(newZ)+1)
						m.Ndvgl.Set(d, s+newV, m.Ndvgl.Get(d, s+newV)+1)
						m.Ndglz.Set(d, newZ, m.Ndglz.Get(d, newZ)+1)
						m.Ndgl.Set(d, m.Ndgl.Get(d)+1)
					} else {
						m.Nloczw.Set(newZ, wd, m.Nloczw.Get(newZ, wd)+1)
						m.Nlocz.Set(newZ, m.Nlocz.Get(newZ)+1)
						m.Ndvloc.Set(d, s+newV, m.Ndvloc.Get(d, s+newV)+1)
						m.Ndvlocz[d][s+newV][newZ] += 1
					}
					m.Ndsv[d][s][newV] += 1
					m.Nds[d][s] += 1
					m.Ndv[d][s+newV] += 1

					m.Vdsn[d][s][w] = newV
					m.Rdsn[d][s][w] = newR
					m.Zdsn[d][s][w] = newZ
				}
			}
		}(d, doc)
	}

	wg.Wait()
}

func NewMGLDA(globalK, localK int, gamma, globalAlpha, localAlpha,
	globalAlphaMix, localAlphaMix, globalBeta, localBeta,
	t, w float64, docs []*Document) *MGLDA {

	docLen := len(docs)
	inflation := float64(0)
	m := MGLDA{
		GlobalK:        globalK,
		LocalK:         localK,
		Gamma:          gamma,
		GlobalAlpha:    globalAlpha,
		LocalAlpha:     localAlpha,
		GlobalAlphaMix: globalAlphaMix,
		LocalAlphaMix:  localAlphaMix,
		GlobalBeta:     globalBeta,
		LocalBeta:      localBeta,
		Docs:           docs,
		T:              t,
		W:              w,
		Inflation:      inflation,

		Nglzw:  matrix.Zeros(globalK, localL),
		Nglz:   matrix.Zeros(globalK, 1),
		Ndglz:  matrix.Zeros(docLen, globalK),
		Ndgl:   matrix.Zeros(docLen, 1),
		Nloczw: matrix.Zeros(globalK, w),
		Nlocz:  matrix.Zeros(localK, 1),

		Ndvgl:  matrix.Numbers(docLen, t+docLen, inflation),
		Ndv:    matrix.Numbers(docLen, t+docLen, inflation),
		Ndvloc: matrix.Numbers(docLen, t+docLen, inflation),
	}

	for d, doc := range docs {
		vd := [][]int{}
		rd := [][]string{}
		zd := [][]int{}
		ndsvd := [][]float64{}
		ndsd := []float64{}

		m.Ndvlocz = append(m.Ndvlocz, matrix.Numbers(docLen, t+docLen, inflation))

		for s, sts := range doc.sentenses {
			vs := []int{}
			rs := []string{}
			zs := []int{}
			for w, word := range sts.words {
				vs = append(vs, rand.Intn(t))

				tp := rand.Intn(2)
				var r string
				var z int
				if tp == 0 {
					r = globalTopic
					z = rand.Intn(globalK)
				} else {
					r = localTopic
					z = rand.Intn(localK)
				}
				rs = append(rs, r)
				zs = append(zs, z)
			}
			vd = append(vd, vs)
			rd = append(rd, rs)
			zd = append(zd, zs)

			ndsvs := []float64{}
			for i := 0; i < t; i++ {
				ndsvs = append(ndsvs, inflation)
			}
			ndsvd = append(ndsvd, ndsvs)
			ndsd = append(ndsd, inflation)
		}
		m.Vdsn = append(m.Vdsn, vd)
		m.Rdsn = append(m.Rdsn, rd)
		m.Zdsn = append(m.Zdsn, zd)
		m.Ndsv = append(m.Ndsv, ndsvd)
		m.Nds = append(m.Nds, ndsd)
	}

	for d, doc := range docs {
		for s, sts := range doc.sentenses {
			for w, wd := range sts.words {
				v := m.Vdsn[d][s][w]
				r := m.Rdsn[d][s][w]
				z := m.Zdsn[d][s][w]
				if r == globalTopic {
					m.Nglzw.Set(z, wd, m.Nglzw.Get(z, wd)+1)
					m.Nglz.Set(z, m.Nglz.Get(z)+1)
					m.Ndvgl.Set(d, s+v, m.Ndvgl.Get(d, s+v)+1)
					m.Ndglz.Set(d, z, m.Ndglz.Get(d, z)+1)
					m.Ndgl.Set(d, m.Ndgl.Get(d)+1)
				} else {
					m.Nloczw.Set(z, wd, m.Nloczw.Get(z, wd)+1)
					m.Nlocz.Set(z, m.Nlocz.Get(z)+1)
					m.Ndvloc.Set(d, s+v, m.Ndvloc.Get(d, s+v)+1)
					m.Ndvlocz[d][s+v][z] += 1
				}
				m.Ndsv[d][s][v] += 1
				m.Nds[d][s] += 1
				m.Ndv[d][s+v] += 1
			}
		}
	}

	return &m
}

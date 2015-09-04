package mglda

import (
	"bufio"
	"fmt"
	"math/rand"
	"sort"
	"sync"

	"github.com/skelterjohn/go.matrix"
)

const (
	topicLimit  = 20
	globalTopic = "gl"
	localTopic  = "loc"
)

var (
	inferenceWG = sync.WaitGroup{}
	topicWG     = sync.WaitGroup{}

	inferenceLock = sync.RWMutex{}
	topicLock     = sync.Mutex{}
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
	LocalAlphaMix  float64
	GlobalBeta     float64
	LocalBeta      float64
	Docs           []*Document
	T              int
	W              int
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
		inferenceWG.Add(1)
		go func(d int, doc *Document) {
			defer inferenceWG.Done()

			for s, sent := range doc.sentenses {
				for w, wd := range sent.words {
					v := m.Vdsn[d][s][w]
					r := m.Rdsn[d][s][w]
					z := m.Zdsn[d][s][w]

					func() {
						inferenceLock.Lock()
						defer inferenceLock.Unlock()
						if r == globalTopic {
							m.Nglzw.Set(z, wd, m.Nglzw.Get(z, wd)-1)
							m.Nglz.Set(z, 1, m.Nglz.Get(z, 1)-1)
							m.Ndvgl.Set(d, s-v, m.Ndvgl.Get(d, s-v)-1)
							m.Ndglz.Set(d, z, m.Ndglz.Get(d, z)-1)
							m.Ndgl.Set(d, 1, m.Ndgl.Get(d, 1)-1)
						} else {
							m.Nloczw.Set(z, wd, m.Nloczw.Get(z, wd)-1)
							m.Nlocz.Set(z, 1, m.Nlocz.Get(z, 1)-1)
							m.Ndvloc.Set(d, s-v, m.Ndvloc.Get(d, s-v)-1)
							m.Ndvlocz[d][s-v][z] -= 1
						}
						m.Ndsv[d][s][v] -= 1
						m.Nds[d][s] -= 1
						m.Ndv.Set(d, s-v, m.Ndv.Get(d, s-v)-1)
					}()

					pvrz := []float64{}
					newVs := []int{}
					newRs := []string{}
					newZs := []int{}
					for vt := 0; vt < m.T; vt++ {
						for zt := 0; zt < m.GlobalK; zt++ {
							newVs = append(newVs, vt)
							newRs = append(newRs, globalTopic)
							newZs = append(newZs, zt)

							func() {
								inferenceLock.RLock()
								defer inferenceLock.RUnlock()
								term1 := (m.Nglzw.Get(zt, wd) + m.GlobalBeta) / (m.Nglz.Get(zt, 1) + float64(m.W)*m.GlobalBeta)
								term2 := (m.Ndsv[d][s][vt] + m.Gamma) / (m.Nds[d][s] + float64(m.T)*m.Gamma)
								term3 := (m.Ndvgl.Get(d, s+vt) + m.GlobalAlpha) / (m.Ndv.Get(d, s+vt) + m.GlobalAlphaMix + m.LocalAlphaMix)
								term4 := (m.Ndglz.Get(d, zt) + m.GlobalAlpha) / (m.Ndgl.Get(d, 1) + float64(m.GlobalK)*m.GlobalAlpha)
								pvrz = append(pvrz, term1*term2*term3*term4)
							}()

						}
						for zt := 0; zt < m.LocalK; zt++ {
							newVs = append(newVs, vt)
							newRs = append(newRs, localTopic)
							newZs = append(newZs, zt)

							func() {
								inferenceLock.RLock()
								defer inferenceLock.RUnlock()
								term1 := (m.Nloczw.Get(zt, wd) + m.LocalBeta) / (m.Nlocz.Get(zt, 1) + float64(m.W)*m.LocalBeta)
								term2 := (m.Ndsv[d][s][vt] + m.Gamma) / (m.Nds[d][s] + float64(m.T)*m.Gamma)
								term3 := (m.Ndvloc.Get(d, s+vt) + m.LocalAlphaMix) / (m.Ndv.Get(d, s+vt) + m.GlobalAlphaMix + m.LocalAlphaMix)
								term4 := (m.Ndvlocz[d][s+vt][zt] + m.LocalAlpha) / (m.Ndvloc.Get(d, s+vt) + float64(m.LocalK)*m.LocalAlpha)
								pvrz = append(pvrz, term1*term2*term3*term4)
							}()
						}
					}

					randIdx := rand.Intn(len(newZs))
					newV := newVs[randIdx]
					newR := newRs[randIdx]
					newZ := newZs[randIdx]

					func() {
						inferenceLock.Lock()
						defer inferenceLock.Unlock()
						// update
						if newR == globalTopic {
							m.Nglzw.Set(newZ, wd, m.Nglzw.Get(newZ, wd)+1)
							m.Nglz.Set(newZ, 1, m.Nglz.Get(newZ, 1)+1)
							m.Ndvgl.Set(d, s+newV, m.Ndvgl.Get(d, s+newV)+1)
							m.Ndglz.Set(d, newZ, m.Ndglz.Get(d, newZ)+1)
							m.Ndgl.Set(d, 1, m.Ndgl.Get(d, 1)+1)
						} else {
							m.Nloczw.Set(newZ, wd, m.Nloczw.Get(newZ, wd)+1)
							m.Nlocz.Set(newZ, 1, m.Nlocz.Get(newZ, 1)+1)
							m.Ndvloc.Set(d, s+newV, m.Ndvloc.Get(d, s+newV)+1)
							m.Ndvlocz[d][s+newV][newZ] += 1
						}
						m.Ndsv[d][s][newV] += 1
						m.Nds[d][s] += 1
						m.Ndv.Set(d, s+newV, m.Ndv.Get(d, s+newV)+1)

						m.Vdsn[d][s][w] = newV
						m.Rdsn[d][s][w] = newR
						m.Zdsn[d][s][w] = newZ
					}()
				}
			}
		}(d, doc)
	}

	inferenceWG.Wait()
}

// WordDist returns a topic word distribution
func (m *MGLDA) WordDist() (*matrix.DenseMatrix, *matrix.DenseMatrix) {
	newNglzw := m.Nglzw.Copy()
	if err := newNglzw.AddDense(matrix.Ones(newNglzw.Rows(), newNglzw.Cols())); err != nil {
		panic(err)
	}
	newNglz := m.Nglz.Copy()
	if err := newNglz.AddDense(matrix.Ones(newNglz.Rows(), newNglz.Cols())); err != nil {
		panic(err)
	}
	newNglz.Scale(float64(0.1))

	newNloczw := m.Nloczw.Copy()
	if err := m.Nloczw.AddDense(matrix.Ones(m.Nloczw.Rows(), m.Nloczw.Cols())); err != nil {
		panic(err)
	}
	newNlocz := m.Nlocz.Copy()

	if err := newNlocz.AddDense(matrix.Ones(m.Nlocz.Rows(), m.Nlocz.Cols())); err != nil {
		panic(err)
	}
	newNlocz.Scale(float64(0.1))

	gl, err := newNglzw.TimesDense(newNglz)
	if err != nil {
		panic(err)
	}
	loc, err := newNloczw.TimesDense(newNlocz)
	if err != nil {
		panic(err)
	}
	return gl, loc
}

func NewMGLDA(globalK, localK int, gamma, globalAlpha, localAlpha,
	globalAlphaMix, localAlphaMix, globalBeta, localBeta float64,
	t, w int, docs []*Document) *MGLDA {

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

		Nglzw:  matrix.Zeros(globalK, w),
		Nglz:   matrix.Zeros(globalK, 1),
		Ndglz:  matrix.Zeros(docLen, globalK),
		Ndgl:   matrix.Zeros(docLen, 1),
		Nloczw: matrix.Zeros(localK, w),
		Nlocz:  matrix.Zeros(localK, 1),

		Ndvgl:  matrix.Numbers(docLen, t+docLen, inflation),
		Ndv:    matrix.Numbers(docLen, t+docLen, inflation),
		Ndvloc: matrix.Numbers(docLen, t+docLen, inflation),
	}

	for _, doc := range docs {
		vd := [][]int{}
		rd := [][]string{}
		zd := [][]int{}
		ndsvd := [][]float64{}
		ndsd := []float64{}

		m.Ndvlocz = append(m.Ndvlocz, matrix.Numbers(docLen, t+docLen, inflation).Arrays())

		for _, sts := range doc.sentenses {
			vs := []int{}
			rs := []string{}
			zs := []int{}
			for _ = range sts.words {
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
					m.Nglz.Set(z, 1, m.Nglz.Get(z, 1)+1)
					m.Ndvgl.Set(d, s+v, m.Ndvgl.Get(d, s+v)+1)
					m.Ndglz.Set(d, z, m.Ndglz.Get(d, z)+1)
					m.Ndgl.Set(d, 1, m.Ndgl.Get(d, 1)+1)
				} else {
					m.Nloczw.Set(z, wd, m.Nloczw.Get(z, wd)+1)
					m.Nlocz.Set(z, 1, m.Nlocz.Get(z, 1)+1)
					m.Ndvloc.Set(d, s+v, m.Ndvloc.Get(d, s+v)+1)
					m.Ndvlocz[d][s+v][z] += 1
				}
				m.Ndsv[d][s][v] += 1
				m.Nds[d][s] += 1
				m.Ndv.Set(d, s+v, m.Ndv.Get(d, s+v)+1)
			}
		}
	}

	return &m
}

func GetWordTopicDist(m *MGLDA, vocabulary []string, wt bufio.Writer) {
	zGlCount := make([]int, m.GlobalK)
	zLocCount := make([]int, m.LocalK)
	wordGlCount := make([]map[int]int, m.GlobalK)
	wordLocCount := make([]map[int]int, m.LocalK)

	for d, doc := range m.Docs {
		topicWG.Add(1)

		go func(d int, doc *Document) {
			for s, sent := range doc.sentenses {
				for w, wd := range sent.words {
					r := m.Rdsn[d][s][w]
					z := m.Zdsn[d][s][w]
					if r == globalTopic {
						topicLock.Lock()
						defer topicLock.Unlock()
						zGlCount[z] += 1
						wordGlCount[z][wd] += 1
					} else {
						topicLock.Lock()
						defer topicLock.Unlock()
						zLocCount[z] += 1
						wordLocCount[z][wd] += 1
					}
				}
			}
		}(d, doc)
	}
	phiGl, phiLoc := m.WordDist()
	for i := 0; i < m.GlobalK; i++ {
		wt.WriteString(fmt.Sprintf("-- global topic: %d (%d words)\n", i, zGlCount[i]))
		for _, w := range topIndice(phiGl.RowCopy(i), topicLimit) {
			wt.WriteString(fmt.Sprintf("%s: %f (%d)\n",
				vocabulary[w], phiGl.Get(i, w),
				wordGlCount[i][w]))
		}
	}
	for i := 0; i < m.LocalK; i++ {
		wt.WriteString(fmt.Sprintf("-- local topic: %d (%d words)\n", i, zLocCount[i]))
		for _, w := range topIndice(phiLoc.RowCopy(i), topicLimit) {
			wt.WriteString(fmt.Sprintf("%s: %f (%d)\n",
				vocabulary[w], phiLoc.Get(i, w),
				wordLocCount[i][w]))
		}
	}

}

func topIndice(array []float64, limit int) []int {
	if len(array)-limit < 0 {
		limit = len(array)
	}
	mp := map[float64]int{}
	for i, a := range array {
		mp[a] = i
	}

	sort.Float64s(array)
	keys := array[len(array)-limit:]

	idx := []int{}
	for _, k := range keys {
		idx = append([]int{mp[k]}, idx...)
	}
	return idx
}

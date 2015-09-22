package mglda

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/suite"
)

var (
	docs = []Document{
		{
			Sentenses: []Sentense{
				{Words: []int{0, 1, 2, 3, 4, 5}},
				{Words: []int{6, 7, 8, 2, 3, 9, 8, 2, 3, 5,
					10, 1, 11, 0, 12, 4, 13, 14, 15, 16}},
				{Words: []int{17, 2, 0, 18, 19, 15, 20, 21,
					22, 23}},
				{Words: []int{22, 24, 25}},
				{Words: []int{26}},
				{Words: []int{27, 28, 1}},
			},
		},
	}
	vocabulary = []string{"company", "money", "email", "telling",
		"product", "shipped", "week", "half", "received",
		"item", "finally", "back", "buy", "wo", "work",
		"phone", "depicts", "numerous", "ca", "find", "number",
		"website", "kind", "response", "customer", "service",
		"problem", "advice", "waste"}
)

type data struct {
	Docs       []Document `json:"docs"`
	Vocabulary []string   `json:"vocabulary"`
}

type MGLDATestSuite struct {
	suite.Suite
	m *MGLDA
}

func (m *MGLDATestSuite) SetupTest() {

}

func TestNewMGLDA(t *testing.T) {
	m := NewMGLDA(4, 2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 3,
		len(vocabulary), &docs)
	assert.Equal(t, m.Inflation, float64(0))
	for i := 0; i < m.Nglzw.Rows(); i++ {
		var sum float64
		for j := 0; j < m.Nglzw.Cols(); j++ {
			sum += m.Nglzw.Get(i, j)
		}
		assert.Equal(t, sum, m.Nglz.Get(i, 0))
	}
	for i := 0; i < m.Nloczw.Rows(); i++ {
		var sum float64
		for j := 0; j < m.Nloczw.Cols(); j++ {
			sum += m.Nloczw.Get(i, j)
		}
		assert.Equal(t, sum, m.Nlocz.Get(i, 0))
	}
}

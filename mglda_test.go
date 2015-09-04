package mglda

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestTopIndice(t *testing.T) {
	assert.Equal(t, []int{}, topIndice([]float64{}, 20))
	assert.Equal(t, []int{0}, topIndice([]float64{1}, 20))
	assert.Equal(t, []int{2, 1}, topIndice([]float64{3, 7, 8}, 2))
	assert.Equal(t, []int{3, 1}, topIndice([]float64{8, 11, 2, 19}, 2))
}

package bit

import (
	"fmt"
)

// Vector is a bitvector.
type Vector struct {
	data   buf
	bitNum int
}

// Vector creates a new bitvector. All bits are initialized with zero.
func NewVector(bitNum int) *Vector {
	return &Vector{
		data:   make(buf, nWords(bitNum, 1)),
		bitNum: bitNum,
	}
}

func (v *Vector) getAsUint64(n int) uint64 {
	return uint64(v.data[n])
}

// Set sets the nth bit one.
func (v *Vector) Set(n int) error {
	if n < 0 || n >= v.bitNum {
		return fmt.Errorf("invalid Vector index: %v", n)
	}

	v.data[n/wordBits] |= 1 << uint(n%wordBits)
	return nil
}

func (v *Vector) reverse(n int) error {
	if n < 0 || n >= v.bitNum {
		return fmt.Errorf("invalid Vector index: %v", n)
	}

	v.data[n/wordBits] ^= 1 << uint(n%wordBits)
	return nil
}

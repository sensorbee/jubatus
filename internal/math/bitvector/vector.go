package bitvector

import (
	"fmt"
)

type Vector struct {
	data   buf
	bitNum int
}

func NewVector(bitNum int) *Vector {
	return &Vector{
		data:   make(buf, nWords(bitNum, 1)),
		bitNum: bitNum,
	}
}

func (v *Vector) getAsUint64(n int) uint64 {
	return uint64(v.data[n])
}

func (v *Vector) Set(n int) error {
	if n < 0 || n >= v.bitNum {
		return fmt.Errorf("invalid Vector index: %v", n)
	}

	v.data[n/wordBits] |= 1 << uint(n%wordBits)
	return nil
}

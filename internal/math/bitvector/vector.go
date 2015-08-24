package bitvector

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

func (v *Vector) GetAsUint64(n int) uint64 {
	return uint64(v.data[n])
}

func (v *Vector) Set(n int) {
	if n < 0 || n >= v.bitNum {
		panic("TODO: fix")
	}

	v.data[n/wordBits] |= 1 << uint(n%wordBits)
}

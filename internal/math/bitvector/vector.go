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

func HammingDistance(x, y *Vector) int {
	minLen := len(y.data)
	maxLen := len(x.data)
	if len(x.data) < len(y.data) {
		x, y = y, x
		minLen, maxLen = maxLen, minLen
	}

	var ret int
	for i := 0; i < minLen; i++ {
		ret += bitcount(x.data[i] ^ y.data[i])
	}
	for i := minLen; i < maxLen; i++ {
		ret += bitcount(x.data[i])
	}
	return ret
}

func bitcount(x word) int {
	var ret int
	for x != 0 {
		ret++
		x &= x - 1
	}
	return ret
}

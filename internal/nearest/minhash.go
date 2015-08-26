package nearest

import (
	"fmt"
	"io"
	"math"
	"pfi/sensorbee/jubatus/internal/math/bitvector"
)

type Minhash struct {
	data bitvector.Array
}

const (
	minhashFormatVersion = 1
)

func NewMinhash(bitNum int) *Minhash {
	return &Minhash{
		data: bitvector.NewArray(bitNum),
	}
}

func (m *Minhash) name() string {
	return "minhash"
}

func (m *Minhash) save(w io.Writer) error {
	if _, err := w.Write([]byte{minhashFormatVersion}); err != nil {
		return err
	}
	return m.data.Save(w)
}

func loadMinhash(r io.Reader) (*Minhash, error) {
	formatVersion := make([]byte, 1)
	if _, err := r.Read(formatVersion); err != nil {
		return nil, err
	}

	switch formatVersion[0] {
	case 1:
		return loadMinhashFormatV1(r)
	default:
		return nil, fmt.Errorf("unsupported format version of minhash container: %v", formatVersion[0])
	}
}

func loadMinhashFormatV1(r io.Reader) (*Minhash, error) {
	data, err := bitvector.LoadArray(r)
	if err != nil {
		return nil, err
	}
	return &Minhash{data}, nil
}

func (m *Minhash) SetRow(id ID, v FeatureVector) {
	if int(id) > m.data.Len() {
		m.data.Resize(int(id))
	}
	m.data.Set(int(id-1), m.hash(v))
}

func (m *Minhash) NeighborRowFromID(id ID, size int) []IDist {
	hash, _ := m.data.Get(int(id - 1))
	return m.neighborRowFromHash(hash, size)
}

func (m *Minhash) NeighborRowFromFV(v FeatureVector, size int) []IDist {
	return m.neighborRowFromHash(m.hash(v), size)
}

func (m *Minhash) neighborRowFromHash(x *bitvector.Vector, size int) []IDist {
	return rankingHammingBitVectors(m.data, x, size)
}

func (m *Minhash) hash(v FeatureVector) *bitvector.Vector {
	bitNum := m.data.BitNum()
	minValues := generateMinValuesBuffer(bitNum)
	hashes := make([]uint64, bitNum)
	for i := range v {
		dim := v[i].Dim
		x := v[i].Value
		keyHash := calcStringHash(dim)
		for j := 0; j < bitNum; j++ {
			hashVal := calcHash(keyHash, uint64(j), x)
			if hashVal < minValues[j] {
				minValues[j] = hashVal
				hashes[j] = keyHash
			}
		}
	}

	bv := bitvector.NewVector(bitNum)
	for i := 0; i < len(hashes); i++ {
		if (hashes[i] & 1) == 1 {
			bv.Set(i)
		}
	}

	return bv
}

func generateMinValuesBuffer(n int) []float32 {
	ret := make([]float32, n)
	for i := 0; i < n; i++ {
		ret[i] = inf32
	}
	return ret
}

var inf32 = float32(math.Inf(1))

func hashMix64(a, b, c uint64) (uint64, uint64, uint64) {
	a -= b
	a -= c
	a ^= (c >> 43)

	b -= c
	b -= a
	b ^= (a << 9)

	c -= a
	c -= b
	c ^= (b >> 8)

	a -= b
	a -= c
	a ^= (c >> 38)

	b -= c
	b -= a
	b ^= (a << 23)

	c -= a
	c -= b
	c ^= (b >> 5)

	a -= b
	a -= c
	a ^= (c >> 35)

	b -= c
	b -= a
	b ^= (a << 49)

	c -= a
	c -= b
	c ^= (b >> 11)

	a -= b
	a -= c
	a ^= (c >> 12)

	b -= c
	b -= a
	b ^= (a << 18)

	c -= a
	c -= b
	c ^= (b >> 22)

	return a, b, c
}

func calcHash(a, b uint64, val float32) float32 {
	const hashPrime = 0xc3a5c85c97cb3127

	var c uint64 = hashPrime
	a, b, c = hashMix64(a, b, c)
	a, b, c = hashMix64(a, b, c)
	r := float32(a) / float32(0xFFFFFFFFFFFFFFFF)
	return -log32(r) / val
}

func log32(x float32) float32 {
	return float32(math.Log(float64(x)))
}

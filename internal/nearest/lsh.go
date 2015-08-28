package nearest

import (
	"fmt"
	"io"
	"pfi/sensorbee/jubatus/internal/math/bit"
)

type LSH struct {
	data bit.Array
}

const (
	lshFormatVersion = 1
)

func NewLSH(bitNum int) *LSH {
	return &LSH{
		data: bit.NewArray(bitNum),
	}
}

func (l *LSH) name() string {
	return "lsh"
}

func (l *LSH) save(w io.Writer) error {
	if _, err := w.Write([]byte{lshFormatVersion}); err != nil {
		return err
	}
	return l.data.Save(w)
}

func loadLSH(r io.Reader) (*LSH, error) {
	formatVersion := make([]byte, 1)
	if _, err := r.Read(formatVersion); err != nil {
		return nil, err
	}

	switch formatVersion[0] {
	case 1:
		return loadLSHFormatV1(r)
	default:
		return nil, fmt.Errorf("unsupported format version of lsh container: %v", formatVersion[0])
	}
}

func loadLSHFormatV1(r io.Reader) (*LSH, error) {
	data, err := bit.LoadArray(r)
	if err != nil {
		return nil, err
	}
	return &LSH{data}, nil
}

func (l *LSH) SetRow(id ID, v FeatureVector) {
	if int(id) > l.data.Len() {
		l.data.Resize(int(id))
	}
	l.data.Set(int(id-1), l.hash(v))
}

func (l *LSH) NeighborRowFromID(id ID, size int) []IDist {
	hash, _ := l.data.Get(int(id - 1))
	return l.neighborRowFromFV(hash, size)
}

func (l *LSH) NeighborRowFromFV(v FeatureVector, size int) []IDist {
	return l.neighborRowFromFV(l.hash(v), size)
}

func (l *LSH) neighborRowFromFV(x *bit.Vector, size int) []IDist {
	return rankingHammingBitVectors(l.data, x, size)
}

func (l *LSH) hash(v FeatureVector) *bit.Vector {
	return cosineLSH(v, l.data.BitNum())
}

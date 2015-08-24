package nearest

import (
	"fmt"
	"io"
	"pfi/sensorbee/jubatus/internal/math/bitvector"
)

type LSH struct {
	data *bitvector.Array
}

const (
	lshFormatVersion = 1
)

func NewLSH(bitNum int) *LSH {
	return &LSH{
		data: bitvector.NewArray(bitNum),
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
	data, err := bitvector.LoadArray(r)
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
	return l.neighborRowFromFV(l.data.Get(int(id-1)), size)
}

func (l *LSH) NeighborRowFromFV(v FeatureVector, size int) []IDist {
	return l.neighborRowFromFV(l.hash(v), size)
}

func (l *LSH) neighborRowFromFV(x *bitvector.Vector, size int) []IDist {
	return rankingHammingBitVectors(l.data, x, size)
}

func (l *LSH) hash(v FeatureVector) *bitvector.Vector {
	return cosineLSH(v, l.data.BitNum())
}

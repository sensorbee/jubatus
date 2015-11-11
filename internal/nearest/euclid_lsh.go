package nearest

import (
	"fmt"
	"github.com/ugorji/go/codec"
	"io"
	"math"
	"pfi/sensorbee/jubatus/internal/math/bit"
)

type EuclidLSH struct {
	lshs  bit.Array
	norms []float32

	cosTable []float32
}

type euclidLSHMsgpack struct {
	_struct struct{} `codec:",toarray"`
	Norms   []float32
}

const (
	euclidLSHFormatVersion = 1
)

func NewEuclidLSH(hashNum int) *EuclidLSH {
	return &EuclidLSH{
		lshs: bit.NewArray(hashNum),

		cosTable: cosTable(hashNum),
	}
}

func (e *EuclidLSH) name() string {
	return "euclid_lsh"
}

func (e *EuclidLSH) save(w io.Writer) error {
	if _, err := w.Write([]byte{euclidLSHFormatVersion}); err != nil {
		return err
	}

	enc := codec.NewEncoder(w, nnMsgpackHandle)
	if err := enc.Encode(&euclidLSHMsgpack{
		Norms: e.norms,
	}); err != nil {
		return err
	}
	return e.lshs.Save(w)
}

func loadEuclidLSH(r io.Reader) (*EuclidLSH, error) {
	formatVersion := make([]byte, 1)
	if _, err := r.Read(formatVersion); err != nil {
		return nil, err
	}

	switch formatVersion[0] {
	case 1:
		return loadEuclidLSHFormatV1(r)
	default:
		return nil, fmt.Errorf("unsupported format version of euclid_lsh container: %v", formatVersion[0])
	}
}

func loadEuclidLSHFormatV1(r io.Reader) (*EuclidLSH, error) {
	var d euclidLSHMsgpack
	dec := codec.NewDecoder(r, nnMsgpackHandle)
	if err := dec.Decode(&d); err != nil {
		return nil, err
	}
	lshs, err := bit.LoadArray(r)
	if err != nil {
		return nil, err
	}
	return &EuclidLSH{
		lshs:  lshs,
		norms: d.Norms,

		cosTable: cosTable(lshs.BitNum()),
	}, nil
}

func (e *EuclidLSH) SetRow(id ID, v FeatureVector) {
	if len(e.norms) < int(id) {
		e.extend(int(id))
	}

	e.lshs.Set(int(id-1), cosineLSH(v, e.lshs.BitNum()))
	e.norms[id-1] = l2Norm(v)
}

func (e *EuclidLSH) NeighborRowFromID(id ID, size int) []IDist {
	lsh, _ := e.lshs.Get(int(id - 1))
	return e.neighborRowFromHash(lsh, e.norms[id-1], size)
}

func (e *EuclidLSH) NeighborRowFromFV(v FeatureVector, size int) []IDist {
	return e.neighborRowFromHash(cosineLSH(v, e.lshs.BitNum()), l2Norm(v), size)
}

func (e *EuclidLSH) neighborRowFromHash(x *bit.Vector, norm float32, size int) []IDist {
	buf := e.lshs.CalcEuclidLSHScoreAndSortPartially(x, norm, e.norms, e.cosTable, size)
	ret := make([]IDist, minInt(size, len(buf)))
	squaredNorm := norm * norm
	for i := 0; i < len(ret); i++ {
		ret[i] = IDist{
			ID:   ID(buf[i].ID),
			Dist: sqrt32(squaredNorm + buf[i].Dist),
		}
	}
	return ret
}

func (e *EuclidLSH) extend(n int) {
	if e.lshs.Len() < n {
		e.lshs.Resize(n)
		if cap(e.norms) >= n {
			e.norms = e.norms[0:n]
		} else {
			newNorms := make([]float32, n, maxInt(2*cap(e.norms), n))
			copy(newNorms, e.norms)
			e.norms = newNorms
		}
	}
}

func l2Norm(v FeatureVector) float32 {
	return sqrt32(squaredL2Norm(v))
}

func squaredL2Norm(v FeatureVector) float32 {
	var ret float32
	for i := range v {
		x := v[i].Value
		ret += x * x
	}
	return ret
}

func sqrt32(x float32) float32 {
	return float32(math.Sqrt(float64(x)))
}

func cosTable(hashNum int) []float32 {
	ret := make([]float32, hashNum+1)
	ret[0] = 1 // cos(0) == 1
	for i := 1; i < hashNum; i++ {
		theta := float64(i) * math.Pi / float64(hashNum)
		ret[i] = float32(math.Cos(theta))
	}
	ret[hashNum] = -1 // cos(pi) == -1
	return ret
}

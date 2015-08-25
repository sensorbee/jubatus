package nearest

import (
	"fmt"
	"github.com/ugorji/go/codec"
	"io"
	"math"
	"pfi/sensorbee/jubatus/internal/math/bitvector"
)

type EuclidLSH struct {
	lshs  *bitvector.Array
	norms []float32

	cosTable []float32
}

type euclidLSHMsgpack struct {
	_struct       struct{} `codec:",toarray"`
	FormatVersion uint8
	Norms         []float32
}

const (
	euclidLSHFormatVersion = 1
)

func NewEuclidLSH(hashNum int) *EuclidLSH {
	return &EuclidLSH{
		lshs: bitvector.NewArray(hashNum),

		cosTable: cosTable(hashNum),
	}
}

func (e *EuclidLSH) name() string {
	return "euclid_lsh"
}

func (e *EuclidLSH) save(w io.Writer) error {
	enc := codec.NewEncoder(w, nnMsgpackHandle)
	if err := enc.Encode(&euclidLSHMsgpack{
		FormatVersion: euclidLSHFormatVersion,
		Norms:         e.norms,
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
	lshs, err := bitvector.LoadArray(r)
	if err != nil {
		return nil, err
	}
	return &EuclidLSH{
		lshs:  lshs,
		norms: d.Norms,
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

func (e *EuclidLSH) neighborRowFromHash(x *bitvector.Vector, norm float32, size int) []IDist {
	buf := make([]IDist, len(e.norms))
	for i := range buf {
		hDist, _ := bitvector.HammingDistance(e.lshs, i, x)
		score := e.norms[i] * (e.norms[i] - 2*norm*e.cosTable[hDist])
		buf[i] = IDist{
			ID:   ID(i + 1),
			Dist: score,
		}
	}
	partialSortByDist(buf, size)
	ret := make([]IDist, minInt(size, len(buf)))
	squaredNorm := norm * norm
	for i := 0; i < len(ret); i++ {
		ret[i] = IDist{
			ID:   buf[i].ID,
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
	ret := make([]float32, hashNum)
	ret[0] = 1 // cos(0) == 1
	for i := 1; i < len(ret); i++ {
		theta := float64(i) * math.Pi / float64(hashNum)
		ret[i] = float32(math.Cos(theta))
	}
	return ret
}

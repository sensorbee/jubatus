package nearest

import (
	"math"
	"math/rand"
	"pfi/sensorbee/jubatus/internal/math/bitvector"
	"sort"
)

type EuclidLSH struct {
	lshs  *bitvector.Array
	norms []float32
	ndata int
}

func NewEuclidLSH(hashNum int) *EuclidLSH {
	return &EuclidLSH{
		lshs: bitvector.NewArray(hashNum),
	}
}

func (e *EuclidLSH) SetRow(id ID, v FeatureVector) {
	e.ndata = maxInt(e.ndata, int(id))
	if len(e.norms) < int(id) {
		e.extend(int(id))
	}

	e.lshs.Set(int(id-1), cosineLSH(v, e.lshs.BitNum()))
	e.norms[id-1] = l2Norm(v)
}

func (e *EuclidLSH) NeighborRowFromID(id ID, size int) []IDist {
	return e.neighborRowFromHash(e.lshs.Get(int(id-1)), e.norms[id-1], size)
}

func (e *EuclidLSH) NeighborRowFromFV(v FeatureVector, size int) []IDist {
	return e.neighborRowFromHash(cosineLSH(v, e.lshs.BitNum()), l2Norm(v), size)
}

func (e *EuclidLSH) neighborRowFromHash(x *bitvector.Vector, norm float32, size int) []IDist {
	buf := make([]IDist, e.ndata)
	for i := 0; i < e.ndata; i++ {
		hDist := e.lshs.HammingDistance(i, x)
		theta := float64(hDist) * math.Pi / float64(e.lshs.BitNum())
		score := e.norms[i] * (e.norms[i] - 2*norm*float32(math.Cos(theta)))
		buf[i] = IDist{
			ID:   ID(i + 1),
			Dist: score,
		}
	}
	sort.Sort(sortByDist(buf))
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

func (e *EuclidLSH) GetAllRows() []ID {
	// TODO: implement
	return nil
}

func (e *EuclidLSH) extend(n int) {
	len := maxInt(2*len(e.norms), n)
	e.lshs.Resize(len)
	norms := make([]float32, len)
	copy(norms, e.norms)
	e.norms = norms
}

func cosineLSH(v FeatureVector, hashNum int) *bitvector.Vector {
	return binarize(randomProjection(v, hashNum))
}

func randomProjection(v FeatureVector, hashNum int) []float32 {
	proj := make([]float32, hashNum)
	for i := range v {
		dim := v[i].Dim
		x := v[i].Value

		seed := calcStringHash(dim)
		src := rand.NewSource(int64(seed))
		r := rand.New(src)
		for j := 0; j < hashNum; j++ {
			proj[j] += x * float32(r.NormFloat64())
		}
	}
	return proj
}

func binarize(proj []float32) *bitvector.Vector {
	ret := bitvector.NewVector(len(proj))
	for i, x := range proj {
		if x > 0 {
			ret.Set(i)
		}
	}
	return ret
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

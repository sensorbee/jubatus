package nearest

import (
	"math/rand"
	"pfi/sensorbee/jubatus/internal/math/bitvector"
	"sort"
)

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

func rankingHammingBitVectors(bva *bitvector.Array, bv *bitvector.Vector, size int) []IDist {
	len := bva.Len()
	buf := make([]IDist, len)
	for i := 0; i < len; i++ {
		dist := bitvector.HammingDistance(bva, i, bv)
		buf[i] = IDist{
			ID:   ID(i + 1),
			Dist: float32(dist),
		}
	}
	sort.Sort(sortByDist(buf))
	ret := make([]IDist, minInt(size, len))
	bitNum := bva.BitNum()
	for i := range ret {
		ret[i] = IDist{
			ID:   buf[i].ID,
			Dist: buf[i].Dist / float32(bitNum),
		}
	}
	return ret
}

type sortByDist []IDist

func (s sortByDist) Len() int {
	return len(s)
}

func (s sortByDist) Less(i, j int) bool {
	return s[i].Dist < s[j].Dist || (s[i].Dist == s[j].Dist && s[i].ID < s[j].ID)
}

func (s sortByDist) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
}

func calcStringHash(s string) uint64 {
	// FNV-1
	var hash uint64 = 14695981039346656037
	for i := 0; i < len(s); i++ {
		hash *= 1099511628211
		hash ^= uint64(s[i])
	}
	return hash
}

func minInt(x, y int) int {
	if x < y {
		return x
	}
	return y
}

func maxInt(x, y int) int {
	if x < y {
		return y
	}
	return x
}

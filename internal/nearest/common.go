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

func rankingHammingBitVectors(bva bitvector.Array, bv *bitvector.Vector, size int) []IDist {
	len := bva.Len()
	buf := make([]IDist, len)
	for i := 0; i < len; i++ {
		dist, _ := bitvector.HammingDistance(bva, i, bv)
		buf[i] = IDist{
			ID:   ID(i + 1),
			Dist: float32(dist),
		}
	}
	partialSortByDist(buf, size)
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
	return less(&s[i], &s[j])
}

func (s sortByDist) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
}

func less(x, y *IDist) bool {
	return x.Dist < y.Dist || (x.Dist == y.Dist && x.ID < y.ID)
}

func pivot(x, y, z *IDist) IDist {
	if less(x, y) {
		// x < y < z
		if less(y, z) {
			return *y
		}
		// x < z < y
		if less(x, z) {
			return *z
		}
		// z < x < y
		return *x
	}

	// y < x < z
	if less(x, z) {
		return *x
	}
	// y < z < x
	if less(y, z) {
		return *z
	}
	// z < y < x
	return *y
}

func partialSortByDist(dists []IDist, n int) {
	for {
		if n <= 64 {
			partialInsertionSort(dists, n)
			return
		}

		len := len(dists)
		switch {
		case len <= 1:
			return
		case len == 2:
			if less(&dists[1], &dists[0]) {
				dists[0], dists[1] = dists[1], dists[0]
			}
			return
		case len <= n:
			sort.Sort(sortByDist(dists))
			return
		}
		pivot := pivot(&dists[0], &dists[len/2], &dists[len-1])
		l := 0
		r := len

		for {
			for less(&dists[l], &pivot) {
				l++
			}
			for less(&pivot, &dists[r-1]) {
				r--
			}
			if r-l <= 1 {
				break
			}
			dists[l], dists[r-1] = dists[r-1], dists[l]
			l++
			r--
		}
		switch {
		case l < n:
			sort.Sort(sortByDist(dists[:l]))
			dists = dists[l:]
			n = n - l
		case l == n:
			sort.Sort(sortByDist(dists[:l]))
			return
		default: // l > n
			dists = dists[:l]
		}
	}
}

func partialInsertionSort(dists []IDist, n int) {
	len := len(dists)
	if len <= 1 {
		return
	}
	if len <= n {
		sort.Sort(sortByDist(dists))
		return
	}
	if n <= 0 {
		return
	}
	if n == 1 {
		maxIx := maxDistsIx(dists)
		dists[0], dists[maxIx] = dists[maxIx], dists[0]
		return
	}

	// n >= 2 and len > n
	max := maxDistsIx(dists[:n])
	back := &dists[n-1]
	dists[max], *back = *back, dists[max]
	for i := n; i < len; i++ {
		current := &dists[i]
		if less(current, back) {
			cand := maxDistsIx(dists[:n-1])
			if less(current, &dists[cand]) {
				dists[cand], *back, *current = *current, dists[cand], *back
			} else {
				*back, *current = *current, *back
			}
		}
	}
	sort.Sort(sortByDist(dists[:n]))
}

func maxDistsIx(dists []IDist) int {
	// len(dists) must >= 1.
	ix := 0
	for i := 1; i < len(dists); i++ {
		if less(&dists[ix], &dists[i]) {
			ix = i
		}
	}
	return ix
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

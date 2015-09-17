package bit

import (
	"sort"
)

type IDist struct {
	ID   ID
	Dist float32
}

type ID uint32

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

func calcEuclidLSHScoresAndSortPartially(a Array, x *Vector, norm float32, norms []float32, cosTable []float32, n int) []IDist {
	buf := make([]IDist, len(norms))
	for i := range buf {
		hDist, _ := a.HammingDistance(i, x)
		score := calcEuclidLSHScore(norms[i], norm, cosTable[hDist])
		buf[i] = IDist{
			ID:   ID(i + 1),
			Dist: score,
		}
	}
	partialSortByDist(buf, n)
	return buf
}

func calcEuclidLSHScore(normI, norm, cos float32) float32 {
	return normI * (normI - 2*norm*cos)
}

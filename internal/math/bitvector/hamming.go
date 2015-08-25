package bitvector

import (
	"fmt"
)

func HammingDistance(a *Array, n int, v *Vector) (int, error) {
	if a.bitNum != v.bitNum {
		return 0, fmt.Errorf("BitNum mismatch: %v, %v", a.bitNum, v.bitNum)
	}
	if n < 0 || n >= a.Len() {
		return 0, fmt.Errorf("invalid Array index: %v", n)
	}

	lbit := n * a.bitNum
	rbit := lbit + a.bitNum
	l := lbit / wordBits
	r := rbit / wordBits

	nRightBits := rbit % wordBits

	if l == r || (l+1 == r && nRightBits == 0) {
		offset := lbit % wordBits
		x := (a.data[l] >> uint(offset)) & leastBits(a.bitNum)
		return bitcount(x ^ word(v.getAsUint64(0))), nil
	}

	if lbit%wordBits == 0 {
		x := a.data[l:]
		ret := 0
		for i := 0; i < r-l; i++ {
			ret += bitcount(x[i] ^ v.data[i])
		}
		if nRightBits != 0 {
			last := r - l
			ret += bitcount((x[last] & leastBits(nRightBits)) ^ v.data[last])
		}
		return ret, nil
	}

	leftOffset := lbit % wordBits
	nLeftBits := (wordBits - leftOffset)
	leftBits := a.data[l] >> uint(leftOffset)
	ret := 0
	nfull := r - (l + 1)
	x := a.data[l+1 : r]
	for i := 0; i < nfull; i++ {
		ret += bitcount(x[i] ^ v.data[i])
	}
	if nRightBits == 0 {
		ret += bitcount(leftBits ^ v.data[nfull])
	} else {
		x := (a.data[r] & leastBits(nRightBits)) | (leftBits << uint(nRightBits))
		ret += bitcount(x ^ v.data[nfull])
		if nLeftBits+nRightBits > wordBits {
			x := leftBits >> uint(wordBits-nRightBits)
			ret += bitcount(x ^ v.data[nfull+1])
		}
	}
	return ret, nil
}

func bitcount(x word) int {
	var ret int
	for x != 0 {
		ret++
		x &= x - 1
	}
	return ret
}

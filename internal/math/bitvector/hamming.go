package bitvector

import (
	"errors"
	"fmt"
)

func HammingDistance(a Array, n int, v *Vector) (int, error) {
	ga, ok := a.(*GeneralArray)
	if !ok {
		return 0, errors.New("HammingDistance is unimplemented for this type of Array.")
	}
	if ga.bitNum != v.bitNum {
		return 0, fmt.Errorf("BitNum mismatch: %v, %v", ga.bitNum, v.bitNum)
	}
	if n < 0 || n >= ga.Len() {
		return 0, fmt.Errorf("invalid Array index: %v", n)
	}

	lbit := n * ga.bitNum
	rbit := lbit + ga.bitNum
	l := lbit / wordBits
	r := rbit / wordBits

	nRightBits := rbit % wordBits

	if l == r || (l+1 == r && nRightBits == 0) {
		offset := lbit % wordBits
		x := (ga.data[l] >> uint(offset)) & leastBits(ga.bitNum)
		return bitcount(x ^ word(v.getAsUint64(0))), nil
	}

	if lbit%wordBits == 0 {
		x := ga.data[l:]
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
	leftBits := ga.data[l] >> uint(leftOffset)
	ret := 0
	nfull := r - (l + 1)
	x := ga.data[l+1 : r]
	for i := 0; i < nfull; i++ {
		ret += bitcount(x[i] ^ v.data[i])
	}
	if nRightBits == 0 {
		ret += bitcount(leftBits ^ v.data[nfull])
	} else {
		x := (ga.data[r] & leastBits(nRightBits)) | (leftBits << uint(nRightBits))
		ret += bitcount(x ^ v.data[nfull])
		if nLeftBits+nRightBits > wordBits {
			x := leftBits >> uint(wordBits-nRightBits)
			ret += bitcount(x ^ v.data[nfull+1])
		}
	}
	return ret, nil
}

func bitcount(x word) int {
	return int(bitcountTable[x&bitcountMask]) + int(bitcountTable[x>>16&bitcountMask]) +
		int(bitcountTable[x>>32&bitcountMask]) + int(bitcountTable[x>>48])
}

const bitcountMask = word(^uint16(0))

var bitcountTable = [bitcountMask + 1]uint8{}

func init() {
	for i := range bitcountTable {
		var cnt uint8
		n := i
		for n != 0 {
			cnt++
			n &= n - 1
		}
		bitcountTable[i] = cnt
	}
}

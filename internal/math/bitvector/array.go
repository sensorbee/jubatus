package bitvector

type Array struct {
	data   buf
	bitNum int
	len    int
}

func NewArray(bitNum int) *Array {
	if bitNum <= 0 {
		return nil
	}

	return &Array{
		bitNum: bitNum,
	}
}

func (a *Array) Resize(n int) {
	a.reserve(n)
	a.len = n
}

func (a *Array) reserve(n int) {
	currCap := a.cap()
	if n <= currCap {
		return
	}

	newCap := maxInt(n, 2*currCap)
	newBuf := make(buf, nWords(a.bitNum, newCap))
	copy(newBuf, a.data)
	a.data = newBuf
}

func (a *Array) Len() int {
	return a.len
}

func (a *Array) cap() int {
	return len(a.data) * wordBits / a.bitNum
}

func (a *Array) BitNum() int {
	return a.bitNum
}

func (a *Array) Get(n int) *Vector {
	if n < 0 || n >= a.len {
		panic("TODO: fix")
	}

	// the nth bitvector is stored in [lbit, rbit).
	lbit := n * a.bitNum
	rbit := lbit + a.bitNum
	l := lbit / wordBits
	r := rbit / wordBits

	// the bitvector is stored in a word.
	if l == r || (l+1 == r && rbit%wordBits == 0) {
		mask := leastBits(a.bitNum)
		shift := word(lbit) % wordBits
		x := (a.data[l] >> shift) & mask
		return &Vector{
			data:   buf{x},
			bitNum: a.bitNum,
		}
	}

	// the bit vector starts from the least bit in a word.
	if lbit%wordBits == 0 {
		retLen := nWords(a.bitNum, 1)
		retBuf := make(buf, retLen)
		copy(retBuf, a.data[l:])

		nTrailingBits := rbit % wordBits
		if nTrailingBits != 0 {
			retBuf[retLen-1] &= leastBits(nTrailingBits)
		}

		return &Vector{
			data:   retBuf,
			bitNum: a.bitNum,
		}
	}

	retLen := nWords(a.bitNum, 1)
	retBuf := make(buf, retLen)

	leftOffset := lbit % wordBits
	nLeftBits := wordBits - leftOffset
	leftBits := a.data[l] >> uint(leftOffset)
	nRightBits := rbit % wordBits
	nTrailingBits := a.bitNum % wordBits
	copy(retBuf, a.data[l+1:r])
	if nRightBits == 0 {
		retBuf[retLen-1] = leftBits
	} else {
		if nLeftBits+nRightBits <= wordBits {
			set(&retBuf[retLen-1], 0, a.data[r], nRightBits)
			set(&retBuf[retLen-1], nRightBits, leftBits, nLeftBits)
		} else {
			set(&retBuf[retLen-2], 0, a.data[r], nRightBits)
			set(&retBuf[retLen-2], nRightBits, leftBits, wordBits-nRightBits)
			set(&retBuf[retLen-1], 0, leftBits>>uint(wordBits-nRightBits), nTrailingBits)
		}
	}

	return &Vector{
		data:   retBuf,
		bitNum: a.bitNum,
	}
}

func (a *Array) Set(n int, v *Vector) {
	if a.bitNum != v.bitNum {
		panic("TODO: fix")
	}
	if n < 0 || n >= a.len {
		panic("TODO: fix")
	}

	// v will be stored in [lbit, rbit).
	lbit := n * a.bitNum
	rbit := lbit + a.bitNum
	l := lbit / wordBits
	r := rbit / wordBits

	// v will be stored in a word.
	if l == r || (l+1 == r && rbit%wordBits == 0) {
		set(&a.data[l], lbit%wordBits, v.data[0], a.bitNum)
		return
	}

	if lbit%wordBits == 0 {
		if rbit%wordBits == 0 {
			copy(a.data[l:], v.data)
			return
		}
		len := len(v.data)
		copy(a.data[l:], v.data[:len-1])
		set(&a.data[r], 0, v.data[len-1], a.bitNum%wordBits)
		return
	}

	copy(a.data[l+1:r], v.data)
	lOffset := lbit % wordBits
	leftNBits := wordBits - lOffset
	rightNBits := rbit % wordBits
	bitNumRes := a.bitNum % wordBits
	len := len(v.data)
	if leftNBits < bitNumRes {
		set(&a.data[r], 0, v.data[len-1], rightNBits)
		set(&a.data[l], lOffset, v.data[len-1]>>uint(rightNBits), leftNBits)
	} else if leftNBits == bitNumRes {
		// this condition means rbit%wordBits == 0
		set(&a.data[l], lOffset, v.data[len-1], leftNBits)
	} else {
		set(&a.data[r], 0, v.data[len-2], rightNBits)
		set(&a.data[l], lOffset, v.data[len-2]>>uint(rightNBits), wordBits-rightNBits)
		set(&a.data[l], wordBits-bitNumRes, v.data[len-1], bitNumRes)
	}
}

func (a *Array) HammingDistance(n int, y *Vector) int {
	if a.bitNum != y.bitNum {
		panic("TODO: fix")
	}

	lbit := n * a.bitNum
	rbit := lbit + a.bitNum
	l := lbit / wordBits
	r := rbit / wordBits

	nRightBits := rbit % wordBits

	if l == r || (l+1 == r && nRightBits == 0) {
		offset := lbit % wordBits
		x := (a.data[l] >> uint(offset)) & leastBits(a.bitNum)
		return bitcount(x ^ word(y.GetAsUint64(0)))
	}

	if lbit%wordBits == 0 {
		x := a.data[l:]
		ret := 0
		for i := 0; i < r-l; i++ {
			ret += bitcount(x[i] ^ y.data[i])
		}
		if nRightBits != 0 {
			last := r - l
			ret += bitcount((x[last] & leastBits(nRightBits)) ^ y.data[last])
		}
		return ret
	}

	leftOffset := lbit % wordBits
	nLeftBits := (wordBits - leftOffset)
	leftBits := a.data[l] >> uint(leftOffset)
	ret := 0
	nfull := r - (l + 1)
	x := a.data[l+1 : r]
	for i := 0; i < nfull; i++ {
		ret += bitcount(x[i] ^ y.data[i])
	}
	if nRightBits == 0 {
		ret += bitcount(leftBits ^ y.data[nfull])
	} else {
		x := (a.data[r] & leastBits(nRightBits)) | (leftBits << uint(nRightBits))
		ret += bitcount(x ^ y.data[nfull])
		if nLeftBits+nRightBits > wordBits {
			x := leftBits >> uint(wordBits-nRightBits)
			ret += bitcount(x ^ y.data[nfull+1])
		}
	}
	return ret
}

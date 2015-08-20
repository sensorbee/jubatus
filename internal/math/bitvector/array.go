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
	copy(retBuf, a.data[l+1:])
	leftOffset := lbit % wordBits
	leftBits := a.data[l] << uint(leftOffset)
	nLeftBits := wordBits - leftOffset
	nTrailingBits := a.bitNum % wordBits
	if nLeftBits <= nTrailingBits {
		nRightBits := nTrailingBits - nLeftBits
		retBuf[retLen-1] &= leastBits(nRightBits)
		set(&retBuf[retLen-1], nRightBits, leftBits, nLeftBits)
	} else {
		nLast2Bits := nLeftBits - nTrailingBits
		set(&retBuf[retLen-2], wordBits-nLast2Bits, leftBits, nLast2Bits)
		retBuf[retLen-1] = leftBits << uint(nLast2Bits)
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
		set(&a.data[l], lOffset, v.data[len-1]<<uint(rightNBits), leftNBits)
	} else if leftNBits == bitNumRes {
		// this condition means rbit%wordBits == 0
		set(&a.data[l], lOffset, v.data[len-1], leftNBits)
	} else {
		set(&a.data[r], 0, v.data[len-2], rightNBits)
		set(&a.data[l], lOffset, v.data[len-2]<<uint(rightNBits), wordBits-rightNBits)
		set(&a.data[l], wordBits-bitNumRes, v.data[len-1], bitNumRes)
	}
}

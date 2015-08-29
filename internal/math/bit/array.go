package bit

import (
	"fmt"
	"github.com/ugorji/go/codec"
	"io"
)

type Array interface {
	Resize(n int)
	Len() int
	BitNum() int
	HammingDistance(int, *Vector) (int, error)
	CalcEuclidLSHScoreAndSortPartially(x *Vector, norm float32, norms []float32, cosTable []float32, n int) []IDist
	Get(int) (*Vector, error)
	Set(int, *Vector) error
	Save(io.Writer) error
}

type largeBitsArray struct {
	data   buf
	bitNum int
	len    int
}

type arrayData struct {
	_struct struct{} `codec",toarray"`
	Data    buf
	BitNum  int
	Len     int
}

func NewArray(bitNum int) Array {
	return createArray(nil, bitNum, 0)
}

func createArray(data buf, bitNum int, len int) Array {
	if bitNum <= 0 || len < 0 {
		return nil
	}

	if bitNum < wordBits {
		// 2^n
		if bitNum&(bitNum-1) == 0 {
			return &smallPowerOfTwoBitsArray{
				data:   data,
				bitNum: bitNum,
				len:    len,
			}
		}
		return &smallBitsArray{
			ga: largeBitsArray{
				data:   data,
				bitNum: bitNum,
				len:    len,
			},
		}
	}

	if bitNum == wordBits {
		return &wordArray{
			data: data,
		}
	}

	if bitNum%wordBits == 0 {
		return &multipleOfWordBitsArray{
			data:   data,
			bitNum: bitNum,
		}
	}

	return &largeBitsArray{
		data:   data,
		bitNum: bitNum,
		len:    len,
	}
}

func (a *largeBitsArray) Resize(n int) {
	a.reserve(n)
	a.len = n
}

func (a *largeBitsArray) reserve(n int) {
	currCap := a.cap()
	if n <= currCap {
		return
	}

	newCap := maxInt(n, 2*currCap)
	newBuf := make(buf, nWords(a.bitNum, newCap))
	copy(newBuf, a.data)
	a.data = newBuf
}

func (a *largeBitsArray) Len() int {
	return a.len
}

func (a *largeBitsArray) cap() int {
	return len(a.data) * wordBits / a.bitNum
}

func (a *largeBitsArray) BitNum() int {
	return a.bitNum
}

func (a *largeBitsArray) HammingDistance(n int, v *Vector) (int, error) {
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

	if lbit%wordBits == 0 {
		x := a.data[l:]
		ret := 0
		for i := 0; i < r-l; i++ {
			ret += bitcount(x[i] ^ v.data[i])
		}
		last := r - l
		ret += bitcount((x[last] & leastBits(nRightBits)) ^ v.data[last])
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

func (a *largeBitsArray) CalcEuclidLSHScoreAndSortPartially(x *Vector, norm float32, norms []float32, cosTable []float32, n int) []IDist {
	return calcEuclidLSHScoresAndSortPartially(a, x, norm, norms, cosTable, n)
}

func (a *largeBitsArray) Get(n int) (*Vector, error) {
	if n < 0 || n >= a.len {
		return nil, fmt.Errorf("invalid Array index: %v", n)
	}

	// the nth bitvector is stored in [lbit, rbit).
	lbit := n * a.bitNum
	rbit := lbit + a.bitNum
	l := lbit / wordBits
	r := rbit / wordBits

	// the bit vector starts from the least bit in a word.
	if lbit%wordBits == 0 {
		retLen := nWords(a.bitNum, 1)
		retBuf := make(buf, retLen)
		copy(retBuf, a.data[l:])

		nTrailingBits := rbit % wordBits
		retBuf[retLen-1] &= leastBits(nTrailingBits)

		return &Vector{
			data:   retBuf,
			bitNum: a.bitNum,
		}, nil
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
	}, nil
}

func (a *largeBitsArray) Set(n int, v *Vector) error {
	if a.bitNum != v.bitNum {
		return fmt.Errorf("BitNum mismatch: %v, %v", a.bitNum, v.bitNum)
	}
	if n < 0 || n >= a.len {
		return fmt.Errorf("invalid Array index: %v", n)
	}

	// v will be stored in [lbit, rbit).
	lbit := n * a.bitNum
	rbit := lbit + a.bitNum
	l := lbit / wordBits
	r := rbit / wordBits

	if lbit%wordBits == 0 {
		len := len(v.data)
		copy(a.data[l:], v.data[:len-1])
		set(&a.data[r], 0, v.data[len-1], a.bitNum%wordBits)
		return nil
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
	return nil
}

const (
	arrayFormatVersion = 1
)

var arrayMsgpackHandle = &codec.MsgpackHandle{
	RawToString: true,
}

func (a *largeBitsArray) Save(w io.Writer) error {
	if _, err := w.Write([]byte{arrayFormatVersion}); err != nil {
		return err
	}

	enc := codec.NewEncoder(w, arrayMsgpackHandle)
	if err := enc.Encode(&arrayData{
		Data:   a.data,
		BitNum: a.bitNum,
		Len:    a.len,
	}); err != nil {
		return err
	}
	return nil
}

func LoadArray(r io.Reader) (Array, error) {
	formatVersion := make([]byte, 1)
	if _, err := r.Read(formatVersion); err != nil {
		return nil, err
	}

	switch formatVersion[0] {
	case 1:
		return loadArrayFormatV1(r)
	default:
		return nil, fmt.Errorf("unsupported format version of Array: %v", formatVersion[0])
	}
}

func loadArrayFormatV1(r io.Reader) (Array, error) {
	var d arrayData
	dec := codec.NewDecoder(r, arrayMsgpackHandle)
	if err := dec.Decode(&d); err != nil {
		return nil, err
	}

	return createArray(d.Data, d.BitNum, d.Len), nil
}

type smallBitsArray struct {
	ga largeBitsArray
}

func (a *smallBitsArray) Resize(n int) {
	a.ga.Resize(n)
}

func (a *smallBitsArray) Len() int {
	return a.ga.Len()
}

func (a *smallBitsArray) BitNum() int {
	return a.ga.BitNum()
}

func (a *smallBitsArray) HammingDistance(n int, v *Vector) (int, error) {
	if a.BitNum() != v.bitNum {
		return 0, fmt.Errorf("BitNum mismatch: %v, %v", a.BitNum(), v.bitNum)
	}
	if n < 0 || n >= a.Len() {
		return 0, fmt.Errorf("invalid Array index: %v", n)
	}

	lbit := n * a.BitNum()
	rbit := lbit + a.BitNum()
	l := lbit / wordBits
	r := rbit / wordBits
	nRightBits := rbit % wordBits
	loffset := uint(lbit % wordBits)

	var x word
	if l == r || nRightBits == 0 {
		x = (a.ga.data[l] >> loffset) & leastBits(a.BitNum())
	} else {
		x = a.ga.data[r] & leastBits(nRightBits)
		x |= (a.ga.data[l] >> loffset) << uint(nRightBits)
	}
	return bitcount(x ^ v.data[0]), nil
}

func (a *smallBitsArray) CalcEuclidLSHScoreAndSortPartially(x *Vector, norm float32, norms []float32, cosTable []float32, n int) []IDist {
	return calcEuclidLSHScoresAndSortPartially(a, x, norm, norms, cosTable, n)
}

func (a *smallBitsArray) Get(n int) (*Vector, error) {
	x, err := a.get(n)
	if err != nil {
		return nil, err
	}
	v := NewVector(a.ga.bitNum)
	v.data[0] = x
	return v, nil
}

func (a *smallBitsArray) get(n int) (word, error) {
	if n < 0 || n >= a.Len() {
		return 0, fmt.Errorf("invalid Array index: %v", n)
	}

	lbit := n * a.BitNum()
	rbit := lbit + a.BitNum()
	l := lbit / wordBits
	r := rbit / wordBits
	nRightBits := rbit % wordBits
	loffset := uint(lbit % wordBits)

	if l == r || nRightBits == 0 {
		x := (a.ga.data[l] >> loffset) & leastBits(a.BitNum())
		return x, nil
	}

	x := a.ga.data[r] & leastBits(nRightBits)
	x |= (a.ga.data[l] >> loffset) << uint(nRightBits)
	return x, nil
}

func (a *smallBitsArray) Set(n int, v *Vector) error {
	if a.ga.bitNum != v.bitNum {
		return fmt.Errorf("BitNum mismatch: %v, %v", a.ga.bitNum, v.bitNum)
	}
	if n < 0 || n >= a.Len() {
		return fmt.Errorf("invalid Array index: %v", n)
	}

	lbit := n * a.BitNum()
	rbit := lbit + a.BitNum()
	l := lbit / wordBits
	r := rbit / wordBits
	nRightBits := rbit % wordBits

	if l == r || nRightBits == 0 {
		set(&a.ga.data[l], lbit%wordBits, v.data[0], a.BitNum())
		return nil
	}

	set(&a.ga.data[r], 0, v.data[0], nRightBits)
	set(&a.ga.data[l], lbit%wordBits, v.data[0]>>uint(nRightBits), a.BitNum()-nRightBits)
	return nil
}

func (a *smallBitsArray) Save(w io.Writer) error {
	return a.ga.Save(w)
}

type wordArray struct {
	data buf
}

func (a *wordArray) Resize(n int) {
	cap := cap(a.data)
	if n <= cap {
		a.data = a.data[:n]
		return
	}
	newBuf := make(buf, n, 2*maxInt(cap, n))
	copy(newBuf, a.data)
	a.data = newBuf
}

func (a *wordArray) Len() int {
	return len(a.data)
}

func (a *wordArray) BitNum() int {
	return wordBits
}

func (a *wordArray) HammingDistance(n int, v *Vector) (int, error) {
	if v.bitNum != wordBits {
		return 0, fmt.Errorf("BitNum mismatch: %v, %v", wordBits, v.bitNum)
	}
	if n < 0 || n >= a.Len() {
		return 0, fmt.Errorf("invalid Array index: %v", n)
	}
	return bitcount(a.data[n] ^ v.data[0]), nil
}

func (a *wordArray) CalcEuclidLSHScoreAndSortPartially(x *Vector, norm float32, norms []float32, cosTable []float32, n int) []IDist {
	buf := make([]IDist, len(norms))
	m := x.data[0]
	for i := range buf {
		hDist := bitcount(a.data[i] ^ m)
		score := calcEuclidLSHScore(norms[i], norm, cosTable[hDist])
		buf[i] = IDist{
			ID:   ID(i + 1),
			Dist: score,
		}
	}
	partialSortByDist(buf, n)
	return buf
}

func (a *wordArray) Get(n int) (*Vector, error) {
	if n < 0 || n >= a.Len() {
		return nil, fmt.Errorf("invalid Array index: %v", n)
	}
	return &Vector{
		data:   a.data[n : n+1],
		bitNum: wordBits,
	}, nil
}

func (a *wordArray) Set(n int, v *Vector) error {
	if v.bitNum != wordBits {
		return fmt.Errorf("BitNum mismatch: %v, %v", wordBits, v.bitNum)
	}
	if n < 0 || n >= a.Len() {
		return fmt.Errorf("invalid Array index: %v", n)
	}
	a.data[n] = v.data[0]
	return nil
}

func (a *wordArray) Save(w io.Writer) error {
	ga := &largeBitsArray{
		data:   a.data,
		bitNum: wordBits,
		len:    len(a.data),
	}
	return ga.Save(w)
}

type smallPowerOfTwoBitsArray struct {
	data   buf
	bitNum int
	len    int
}

func (a *smallPowerOfTwoBitsArray) Resize(n int) {
	newDataLen := nWords(a.bitNum, n)
	cap := len(a.data)
	if cap >= newDataLen {
		a.len = n
		return
	}
	newBuf := make(buf, 2*maxInt(cap, newDataLen))
	copy(newBuf, a.data)
	a.data = newBuf
	a.len = n
}

func (a *smallPowerOfTwoBitsArray) Len() int {
	return a.len
}

func (a *smallPowerOfTwoBitsArray) BitNum() int {
	return a.bitNum
}

func (a *smallPowerOfTwoBitsArray) HammingDistance(n int, v *Vector) (int, error) {
	if a.bitNum != v.bitNum {
		return 0, fmt.Errorf("BitNum mismatch: %v, %v", a.bitNum, v.bitNum)
	}
	if n < 0 || n >= a.Len() {
		return 0, fmt.Errorf("invalid Array index: %v", n)
	}

	switch a.bitNum {
	case 1:
		full := a.data[n/64]
		part := (full >> uint(n%64)) & 1
		return int(part ^ v.data[0]), nil
	case 2:
		full := a.data[n/32]
		part := (full >> uint(2*(n%32))) & 3
		return bitcount8(uint8(part ^ v.data[0])), nil
	case 4:
		full := a.data[n/16]
		part := (full >> uint(4*(n%16))) & 0xF
		return bitcount8(uint8(part ^ v.data[0])), nil
	case 8:
		full := a.data[n/8]
		part := (full >> uint(8*(n%8))) & 0xFF
		return bitcount8(uint8(part ^ v.data[0])), nil
	case 16:
		full := a.data[n/4]
		part := uint16(full >> uint(16*(n%4)))
		return bitcount16(part ^ uint16(v.data[0])), nil
	case 32:
		full := a.data[n/2]
		part := uint32(full >> uint(32*(n%2)))
		return bitcount32(part ^ uint32(v.data[0])), nil
	}
	return 0, fmt.Errorf("invalid BitNum: %v", a.bitNum)
}

func (a *smallPowerOfTwoBitsArray) CalcEuclidLSHScoreAndSortPartially(x *Vector, norm float32, norms []float32, cosTable []float32, n int) []IDist {
	if a.bitNum == 32 {
		buf := make([]IDist, len(norms))
		m := x.data[0] | x.data[0]<<32
		for i := 0; i < a.len/2; i++ {
			ix1, ix2 := 2*i, 2*i+1
			hDist1, hDist2 := bitcount32s(uint64(a.data[i] ^ m))
			score1 := calcEuclidLSHScore(norms[ix1], norm, cosTable[hDist1])
			score2 := calcEuclidLSHScore(norms[ix2], norm, cosTable[hDist2])
			buf[ix1] = IDist{
				ID:   ID(ix1 + 1),
				Dist: score1,
			}
			buf[ix2] = IDist{
				ID:   ID(ix2 + 1),
				Dist: score2,
			}
		}
		if a.len%2 == 1 {
			hDist := bitcount32(uint32(a.data[a.len/2] ^ m))
			score := norms[a.len-1] * (norms[a.len-1] - 2*norm*cosTable[hDist])
			buf[a.len-1] = IDist{
				ID:   ID(a.len),
				Dist: score,
			}
		}
		partialSortByDist(buf, n)
		return buf
	}

	return calcEuclidLSHScoresAndSortPartially(a, x, norm, norms, cosTable, n)
}

func (a *smallPowerOfTwoBitsArray) Get(n int) (*Vector, error) {
	if n < 0 || n >= a.Len() {
		return nil, fmt.Errorf("invalid Array index: %v", n)
	}
	v := NewVector(a.bitNum)
	v.data[0] = a.get(n)
	return v, nil
}

func (a *smallPowerOfTwoBitsArray) get(n int) word {
	nelems := wordBits / a.bitNum
	mask := leastBits(a.bitNum)
	return (a.data[n/nelems] >> uint(n%nelems*a.bitNum)) & mask
}

func (a *smallPowerOfTwoBitsArray) Set(n int, v *Vector) error {
	if a.bitNum != v.bitNum {
		return fmt.Errorf("BitNum mismatch: %v, %v", a.bitNum, v.bitNum)
	}
	if n < 0 || n >= a.Len() {
		return fmt.Errorf("invalid Array index: %v", n)
	}
	nelems := wordBits / a.bitNum
	mask := leastBits(a.bitNum)
	offset := uint(n % nelems * a.bitNum)
	a.data[n/nelems] &= ^(mask << offset)
	a.data[n/nelems] |= v.data[0] << offset
	return nil
}

func (a *smallPowerOfTwoBitsArray) Save(w io.Writer) error {
	ga := &largeBitsArray{
		data:   a.data,
		bitNum: a.bitNum,
		len:    a.len,
	}
	return ga.Save(w)
}

type multipleOfWordBitsArray struct {
	data   buf
	bitNum int
}

func (a *multipleOfWordBitsArray) Resize(n int) {
	newLen := n * (a.bitNum / wordBits)
	cap := cap(a.data)
	if cap >= newLen {
		a.data = a.data[:newLen]
		return
	}
	newBuf := make(buf, newLen, 2*maxInt(cap, newLen))
	copy(newBuf, a.data)
	a.data = newBuf
}

func (a *multipleOfWordBitsArray) Len() int {
	return len(a.data) / (a.bitNum / wordBits)
}

func (a *multipleOfWordBitsArray) BitNum() int {
	return a.bitNum
}

func (a *multipleOfWordBitsArray) HammingDistance(n int, v *Vector) (int, error) {
	if a.bitNum != v.bitNum {
		return 0, fmt.Errorf("BitNum mismatch: %v, %v", a.bitNum, v.bitNum)
	}
	// omit n >= a.Len() because a.Len() is slow.
	if n < 0 {
		return 0, fmt.Errorf("invalid Array index: %v", n)
	}
	nw := a.bitNum / wordBits
	var ret int
	for i := 0; i < nw; i++ {
		ret += bitcount(a.data[n*nw+i] ^ v.data[i])
	}
	return ret, nil
}

func (a *multipleOfWordBitsArray) CalcEuclidLSHScoreAndSortPartially(x *Vector, norm float32, norms []float32, cosTable []float32, n int) []IDist {
	buf := make([]IDist, len(norms))
	nWords := a.bitNum / wordBits
	for i := range buf {
		var hDist int
		for j := range x.data {
			hDist += bitcount(a.data[nWords*i+j] ^ x.data[j])
		}
		score := calcEuclidLSHScore(norms[i], norm, cosTable[hDist])
		buf[i] = IDist{
			ID:   ID(i + 1),
			Dist: score,
		}
	}
	partialSortByDist(buf, n)
	return buf
}

func (a *multipleOfWordBitsArray) Get(n int) (*Vector, error) {
	if n < 0 || n >= a.Len() {
		return nil, fmt.Errorf("invalid Array index: %v", n)
	}
	v := NewVector(a.bitNum)
	nw := a.bitNum / wordBits
	copy(v.data, a.data[n*nw:])
	return v, nil
}

func (a *multipleOfWordBitsArray) Set(n int, v *Vector) error {
	if a.bitNum != v.bitNum {
		return fmt.Errorf("BitNum mismatch: %v, %v", a.bitNum, v.bitNum)
	}
	if n < 0 || n >= a.Len() {
		return fmt.Errorf("invalid Array index: %v", n)
	}
	nw := a.bitNum / wordBits
	copy(a.data[n*nw:], v.data)
	return nil
}

func (a *multipleOfWordBitsArray) Save(w io.Writer) error {
	ga := &largeBitsArray{
		data:   a.data,
		bitNum: a.bitNum,
		len:    a.Len(),
	}
	return ga.Save(w)
}

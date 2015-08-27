package bitvector

import (
	"errors"
	"fmt"
	"github.com/ugorji/go/codec"
	"io"
)

type Array interface {
	Resize(n int)
	Len() int
	BitNum() int
	HammingDistance(int, *Vector) (int, error)
	Get(int) (*Vector, error)
	Set(int, *Vector) error
	Save(io.Writer) error
}

type GeneralArray struct {
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
	if bitNum <= 0 {
		return nil
	}

	if bitNum == wordBits {
		return &WordArray{}
	}

	// 2^n
	if bitNum&(bitNum-1) == 0 {
		if bitNum < wordBits {
			return &SmallPowerOfTwoArray{
				bitNum: bitNum,
			}
		}
	}

	if bitNum%wordBits == 0 {
		return &MultileOfWordBitsArray{
			bitNum: bitNum,
		}
	}

	return &GeneralArray{
		bitNum: bitNum,
	}
}

func (a *GeneralArray) Resize(n int) {
	a.reserve(n)
	a.len = n
}

func (a *GeneralArray) reserve(n int) {
	currCap := a.cap()
	if n <= currCap {
		return
	}

	newCap := maxInt(n, 2*currCap)
	newBuf := make(buf, nWords(a.bitNum, newCap))
	copy(newBuf, a.data)
	a.data = newBuf
}

func (a *GeneralArray) Len() int {
	return a.len
}

func (a *GeneralArray) cap() int {
	return len(a.data) * wordBits / a.bitNum
}

func (a *GeneralArray) BitNum() int {
	return a.bitNum
}

func (a *GeneralArray) HammingDistance(n int, v *Vector) (int, error) {
	return HammingDistance(a, n, v)
}

func (a *GeneralArray) Get(n int) (*Vector, error) {
	if n < 0 || n >= a.len {
		return nil, fmt.Errorf("invalid Array index: %v", n)
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
		}, nil
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

func (a *GeneralArray) Set(n int, v *Vector) error {
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

	// v will be stored in a word.
	if l == r || (l+1 == r && rbit%wordBits == 0) {
		set(&a.data[l], lbit%wordBits, v.data[0], a.bitNum)
		return nil
	}

	if lbit%wordBits == 0 {
		if rbit%wordBits == 0 {
			copy(a.data[l:], v.data)
			return nil
		}
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

func (a *GeneralArray) Save(w io.Writer) error {
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
	return &GeneralArray{
		data:   d.Data,
		bitNum: d.BitNum,
		len:    d.Len,
	}, nil
}

type WordArray struct {
	data buf
}

func (a *WordArray) Resize(n int) {
	cap := cap(a.data)
	if n <= cap {
		a.data = a.data[:n]
		return
	}
	newBuf := make(buf, n, 2*maxInt(cap, n))
	copy(newBuf, a.data)
	a.data = newBuf
}

func (a *WordArray) Len() int {
	return len(a.data)
}

func (a *WordArray) BitNum() int {
	return wordBits
}

func (a *WordArray) HammingDistance(n int, v *Vector) (int, error) {
	if v.bitNum != wordBits {
		return 0, fmt.Errorf("BitNum mismatch: %v, %v", wordBits, v.bitNum)
	}
	if n < 0 || n >= a.Len() {
		return 0, fmt.Errorf("invalid Array index: %v", n)
	}
	return bitcount(a.data[n] ^ v.data[0]), nil
}

func (a *WordArray) Get(n int) (*Vector, error) {
	if n < 0 || n >= a.Len() {
		return nil, fmt.Errorf("invalid Array index: %v", n)
	}
	return &Vector{
		data:   a.data[n : n+1],
		bitNum: wordBits,
	}, nil
}

func (a *WordArray) Set(n int, v *Vector) error {
	if v.bitNum != wordBits {
		return fmt.Errorf("BitNum mismatch: %v, %v", wordBits, v.bitNum)
	}
	if n < 0 || n >= a.Len() {
		return fmt.Errorf("invalid Array index: %v", n)
	}
	a.data[n] = v.data[0]
	return nil
}

func (a *WordArray) Save(io.Writer) error {
	return errors.New("TODO: implement")
}

type SmallPowerOfTwoArray struct {
	data   buf
	bitNum int
	len    int
}

func (a *SmallPowerOfTwoArray) Resize(n int) {
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

func (a *SmallPowerOfTwoArray) Len() int {
	return a.len
}

func (a *SmallPowerOfTwoArray) BitNum() int {
	return a.bitNum
}

func (a *SmallPowerOfTwoArray) HammingDistance(n int, v *Vector) (int, error) {
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
		return bitcount16(uint16(part ^ v.data[0])), nil
	case 4:
		full := a.data[n/16]
		part := (full >> uint(4*(n%16))) & 0xF
		return bitcount16(uint16(part ^ v.data[0])), nil
	case 8:
		full := a.data[n/8]
		part := (full >> uint(8*(n%8))) & 0xFF
		return bitcount16(uint16(part ^ v.data[0])), nil
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

func (a *SmallPowerOfTwoArray) Get(n int) (*Vector, error) {
	if n < 0 || n >= a.Len() {
		return nil, fmt.Errorf("invalid Array index: %v", n)
	}
	v := NewVector(a.bitNum)
	v.data[0] = a.get(n)
	return v, nil
}

func (a *SmallPowerOfTwoArray) get(n int) word {
	nelems := wordBits / a.bitNum
	mask := leastBits(a.bitNum)
	return (a.data[n/nelems] >> uint(n%nelems*a.bitNum)) & mask
}

func (a *SmallPowerOfTwoArray) Set(n int, v *Vector) error {
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

func (a *SmallPowerOfTwoArray) Save(io.Writer) error {
	return errors.New("TODO: implement")
}

type MultileOfWordBitsArray struct {
	data   buf
	bitNum int
}

func (a *MultileOfWordBitsArray) Resize(n int) {
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

func (a *MultileOfWordBitsArray) Len() int {
	return len(a.data) / (a.bitNum / wordBits)
}

func (a *MultileOfWordBitsArray) BitNum() int {
	return a.bitNum
}

func (a *MultileOfWordBitsArray) HammingDistance(n int, v *Vector) (int, error) {
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

func (a *MultileOfWordBitsArray) Get(n int) (*Vector, error) {
	if n < 0 || n >= a.Len() {
		return nil, fmt.Errorf("invalid Array index: %v", n)
	}
	v := NewVector(a.bitNum)
	nw := a.bitNum / wordBits
	copy(v.data, a.data[n*nw:])
	return v, nil
}

func (a *MultileOfWordBitsArray) Set(n int, v *Vector) error {
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

func (a *MultileOfWordBitsArray) Save(io.Writer) error {
	return errors.New("TODO: implement")
}

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

package nearest

import (
	"fmt"
	"github.com/ugorji/go/codec"
	"io"
	"reflect"
)

type Neighbor interface {
	SetRow(id ID, v FeatureVector)
	NeighborRowFromID(id ID, size int) []IDist
	NeighborRowFromFV(v FeatureVector, size int) []IDist

	name() string
	save(w io.Writer) error
}

type FeatureElement struct {
	Dim   string
	Value float32
}
type FeatureVector []FeatureElement

type IDist struct {
	ID   ID
	Dist float32
}

type ID uint32

var (
	nnMsgpackHandle = &codec.MsgpackHandle{
		RawToString: true,
	}
)

func init() {
	nnMsgpackHandle.MapType = reflect.TypeOf(map[string]interface{}{})
}

const (
	nnFormatVersion = 1
)

type nnMsgpack struct {
	_struct   struct{} `codec:",toarray"`
	Algorithm string
}

func Save(n Neighbor, w io.Writer) error {
	if _, err := w.Write([]byte{nnFormatVersion}); err != nil {
		return err
	}

	enc := codec.NewEncoder(w, nnMsgpackHandle)
	if err := enc.Encode(&nnMsgpack{
		Algorithm: n.name(),
	}); err != nil {
		return err
	}

	return n.save(w)
}

func Load(r io.Reader) (Neighbor, error) {
	formatVersion := make([]byte, 1)
	if _, err := r.Read(formatVersion); err != nil {
		return nil, err
	}

	switch formatVersion[0] {
	case 1:
		return loadFormatV1(r)
	default:
		return nil, fmt.Errorf("unsupported format version of nearest neighbor container: %v", formatVersion[0])
	}
}

func loadFormatV1(r io.Reader) (Neighbor, error) {
	var d nnMsgpack
	dec := codec.NewDecoder(r, nnMsgpackHandle)
	if err := dec.Decode(&d); err != nil {
		return nil, err
	}

	switch d.Algorithm {
	case "lsh":
		return loadLSH(r)
	case "minhash":
		return loadMinhash(r)
	case "euclid_lsh":
		return loadEuclidLSH(r)
	default:
		return nil, fmt.Errorf("unsupported nearest neighbor algorithm: %v", d.Algorithm)
	}
}

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

type ID int64

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
	_struct       struct{} `codec:",toarray"`
	FormatVersion uint8
	Algorithm     string
}

func Save(n Neighbor, w io.Writer) error {
	enc := codec.NewEncoder(w, nnMsgpackHandle)
	if err := enc.Encode(&nnMsgpack{
		FormatVersion: nnFormatVersion,
		Algorithm:     n.name(),
	}); err != nil {
		return err
	}

	return n.save(w)
}

func Load(r io.Reader) (Neighbor, error) {
	var d nnMsgpack
	dec := codec.NewDecoder(r, nnMsgpackHandle)
	if err := dec.Decode(&d); err != nil {
		return nil, err
	}

	switch d.FormatVersion {
	case 1:
		return loadFormatV1(r, d.Algorithm)
	default:
		return nil, fmt.Errorf("unsupported format version of nearest neighbor container: %v", d.FormatVersion)
	}
}

func loadFormatV1(r io.Reader, algo string) (Neighbor, error) {
	switch algo {
	case "lsh":
		return nil, fmt.Errorf("LSH is unimplemented")
	case "minhash":
		return loadMinhash(r)
	case "euclid_lsh":
		return loadEuclidLSH(r)
	default:
		return nil, fmt.Errorf("unsupported nearest neighbor algorithm: %v", algo)
	}
}

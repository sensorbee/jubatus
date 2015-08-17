package intern

import (
	"fmt"
	"github.com/ugorji/go/codec"
	"io"
	"reflect"
)

// Intern is a mapping from strings to ints. It isn't thread safe. Appropriate
// handling of race conditions is the responsibility of users.
type Intern struct {
	storage map[string]int
	gen     int
}

type internData struct {
	_struct struct{} `codec:",toarray"`
	Storage map[string]int
	Gen     int
}

// New creates a new Intern instance.
func New() *Intern {
	return &Intern{
		storage: make(map[string]int),
		gen:     0,
	}
}

// GetOrZero returns an ID for a string if the string is already registered.
// If the string is not registered this method returns zero,
func (i *Intern) GetOrZero(s string) int {
	return i.storage[s]
}

// Get returns an ID for a string. If the string was not registered, this
// method registers the string and returns an ID. This method is idempotent.
func (i *Intern) Get(s string) int {
	id := i.GetOrZero(s)
	if id == 0 {
		i.gen++
		i.storage[s] = i.gen
		return i.gen
	}
	return id
}

const (
	internFormatVersion uint8 = 1
)

var internMsgpackHandle = &codec.MsgpackHandle{
	RawToString: true,
}

func init() {
	internMsgpackHandle.MapType = reflect.TypeOf(map[string]interface{}{})
}

// Save saves the current state of Intern.
func (i *Intern) Save(w io.Writer) error {
	if _, err := w.Write([]byte{internFormatVersion}); err != nil {
		return err
	}

	enc := codec.NewEncoder(w, internMsgpackHandle)
	if err := enc.Encode(&internData{
		Storage: i.storage,
		Gen:     i.gen,
	}); err != nil {
		return err
	}
	return nil
}

// Load loads the saved data to Intern.
func Load(r io.Reader) (*Intern, error) {
	formatVersion := make([]byte, 1)
	if _, err := r.Read(formatVersion); err != nil {
		return nil, err
	}

	switch formatVersion[0] {
	case 1:
		return loadFormatV1(r)
	default:
		return nil, fmt.Errorf("unsupported format version of Intern container: %v", formatVersion[0])
	}
}

func loadFormatV1(r io.Reader) (*Intern, error) {
	var d internData
	dec := codec.NewDecoder(r, internMsgpackHandle)
	if err := dec.Decode(&d); err != nil {
		return nil, err
	}
	return &Intern{
		storage: d.Storage,
		gen:     d.Gen,
	}, nil
}

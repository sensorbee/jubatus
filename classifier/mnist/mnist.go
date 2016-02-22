package mnist

import (
	"bufio"
	"errors"
	"fmt"
	"github.com/sensorbee/jubatus/internal/perline"
	"gopkg.in/sensorbee/sensorbee.v0/bql"
	"gopkg.in/sensorbee/sensorbee.v0/core"
	"gopkg.in/sensorbee/sensorbee.v0/data"
	"io"
	"os"
	"strconv"
	"strings"
)

type source struct {
	path string
}

func (s *source) Parse(line string, lineNo int) (data.Map, error) {
	line = strings.TrimSpace(line)
	fields := strings.Split(line, " ")
	if len(fields) == 0 {
		return nil, perline.Pass
	}
	label := fields[0]
	fv := make(data.Map)
	for _, field := range fields[1:] {
		ix := strings.Index(field, ":")
		if ix < 0 {
			return nil, fmt.Errorf("invalid libsvm format at %s:%d", s.path, lineNo)
		}
		v, err := strconv.ParseFloat(field[ix+1:], 32)
		if err != nil {
			return nil, fmt.Errorf("%v at %s:%d", err, s.path, lineNo)
		}
		fv[field[:ix]] = data.Float(v) / 255
	}
	return data.Map{
		"label":          data.String(label),
		"feature_vector": fv,
	}, nil
}

func (s *source) Reader() (*bufio.Reader, io.Closer, error) {
	f, err := os.Open(s.path)
	if err != nil {
		return nil, nil, err
	}
	r := bufio.NewReader(f)
	return r, f, nil
}

func createSource(ctx *core.Context, ioParams *bql.IOParams, params data.Map) (core.Source, error) {
	v, ok := params["path"]
	if !ok {
		return nil, errors.New("path parameter is missing")
	}
	path, err := data.AsString(v)
	if err != nil {
		return nil, fmt.Errorf("path parameter is not a string: %v", err)
	}

	s := &source{path}
	return core.NewRewindableSource(perline.NewSource(s, s, 1)), nil
}

func init() {
	bql.MustRegisterGlobalSourceCreator("mnist_source", bql.SourceCreatorFunc(createSource))
}

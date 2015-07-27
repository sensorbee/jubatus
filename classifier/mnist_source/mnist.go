package mnist_source

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"os"
	"pfi/sensorbee/sensorbee/bql"
	"pfi/sensorbee/sensorbee/core"
	"pfi/sensorbee/sensorbee/data"
	"strconv"
	"strings"
	"time"
)

type mnistSource struct {
	file *os.File
	r    *bufio.Reader
}

func (m *mnistSource) GenerateStream(ctx *core.Context, w core.Writer) error {
	defer m.file.Close()
	for {
		l, _, err := m.r.ReadLine()
		if err != nil {
			if err == io.EOF {
				return nil
			}
			return err
		}
		line := string(l)
		line = strings.Trim(line, "\n ")
		fields := strings.Split(line, " ")
		if len(fields) <= 1 {
			continue
		}
		label := fields[0]
		fv := make(data.Map)
		for _, s := range fields[1:] {
			pair := strings.Split(s, ":")
			if len(pair) != 2 {
				continue
			}
			v, err := strconv.Atoi(pair[1])
			if err != nil {
				panic(err)
				continue
			}
			fv[pair[0]] = data.Float(v) / 255
		}
		now := time.Now()
		w.Write(ctx, &core.Tuple{
			Data: data.Map{
				"label":          data.String(label),
				"feature_vector": fv,
			},
			Timestamp:     now,
			ProcTimestamp: now,
		})
	}
	return nil
}

func (*mnistSource) Stop(*core.Context) error {
	return nil
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
	return new(path)
}

func new(path string) (*mnistSource, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	r := bufio.NewReaderSize(f, 10000)
	return &mnistSource{f, r}, nil
}

func init() {
	bql.MustRegisterGlobalSourceCreator("mnist_source", bql.SourceCreatorFunc(createSource))
}

package mnist

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

type source struct {
	path string
}

func (s *source) GenerateStream(ctx *core.Context, w core.Writer) error {
	f, err := os.Open(s.path)
	if err != nil {
		return err
	}
	defer f.Close()
	r := bufio.NewReader(f)
	for {
		line, err := r.ReadString('\n')
		if err != nil {
			if err != io.EOF {
				return err
			}
			line = strings.TrimSpace(line)
			if len(line) == 0 {
				return nil
			}
		}
		line = strings.TrimSpace(line)
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
				continue
			}
			fv[pair[0]] = data.Float(v) / 255
		}
		now := time.Now()
		err = w.Write(ctx, &core.Tuple{
			Data: data.Map{
				"label":          data.String(label),
				"feature_vector": fv,
			},
			Timestamp:     now,
			ProcTimestamp: now,
		})
		if err != nil {
			return err
		}
	}
	return nil
}

func (s *source) Stop(*core.Context) error {
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
	return core.NewRewindableSource(&source{
		path: path,
	}), nil
}

func init() {
	bql.MustRegisterGlobalSourceCreator("mnist_source", bql.SourceCreatorFunc(createSource))
}

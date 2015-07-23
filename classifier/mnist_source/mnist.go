package mnist_source

import (
	"bufio"
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
	r *bufio.Reader
}

func (m *mnistSource) GenerateStream(ctx *core.Context, w core.Writer) error {
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

func createTrainingSource(*core.Context, *bql.IOParams, data.Map) (core.Source, error) {
	r, err := newReader("mnist")
	if err != nil {
		return nil, err
	}
	return &mnistSource{r}, nil
}

func createTestSource(*core.Context, *bql.IOParams, data.Map) (core.Source, error) {
	r, err := newReader("mnist.t")
	if err != nil {
		return nil, err
	}
	return &mnistSource{r}, nil
}

func newReader(path string) (*bufio.Reader, error) {
	// does not close because this code is for test and will be removed soon.
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	return bufio.NewReaderSize(f, 10000), nil
}

func init() {
	bql.MustRegisterGlobalSourceCreator("mnist_training", bql.SourceCreatorFunc(createTrainingSource))
	bql.MustRegisterGlobalSourceCreator("mnist_test", bql.SourceCreatorFunc(createTestSource))
}

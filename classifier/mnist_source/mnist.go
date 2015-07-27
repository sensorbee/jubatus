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

func createTrainingSource(*core.Context, *bql.IOParams, data.Map) (core.Source, error) {
	return new("mnist")
}

func createTestSource(*core.Context, *bql.IOParams, data.Map) (core.Source, error) {
	return new("mnist.t")
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
	bql.MustRegisterGlobalSourceCreator("mnist_training", bql.SourceCreatorFunc(createTrainingSource))
	bql.MustRegisterGlobalSourceCreator("mnist_test", bql.SourceCreatorFunc(createTestSource))
}

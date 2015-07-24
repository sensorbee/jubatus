package rent_source

import (
	"bufio"
	"io"
	"os"
	"pfi/sensorbee/sensorbee/bql"
	"pfi/sensorbee/sensorbee/core"
	"pfi/sensorbee/sensorbee/data"
	"strings"
	"time"
)

// https://github.com/jubatus/jubatus-example/tree/master/rent

// TODO: remove panics

type rentSource struct {
	file     *os.File
	r        *bufio.Reader
	training bool
}

func (r *rentSource) GenerateStream(ctx *core.Context, w core.Writer) error {
	numFieldNames := []string{
		"家賃(万円)",
		"駅からの徒歩時間 (分)",
		"専有面積 (m*m)",
		"築年数 (年)",
		"階数",
	}

	defer r.file.Close()
	if r.training {
		for {
			line, err := r.readLine()
			if err != nil {
				if err == io.EOF {
					return nil
				}
				return err
			}
			if line[0] == '#' {
				continue
			}
			fields := strings.Split(line, ", ")
			if len(fields) != 6 {
				panic("hoge")
			}
			value, err := data.ToFloat(data.String(fields[0]))
			if err != nil {
				panic(err)
			}
			fv := make(data.Map)
			for i := 1; i < len(numFieldNames); i++ {
				x, err := data.ToFloat(data.String(fields[i]))
				if err != nil {
					panic(err)
				}
				fv[numFieldNames[i]] = data.Float(x)
			}
			fv[fields[len(fields)-1]] = data.Float(1)
			now := time.Now()
			w.Write(ctx, &core.Tuple{
				Data: data.Map{
					"value":          data.Float(value),
					"feature_vector": fv,
				},
				Timestamp:     now,
				ProcTimestamp: now,
			})
		}
	} else {
		fv := make(data.Map)
		i := 1
		for {
			line, err := r.readLine()
			if err != nil {
				if err == io.EOF {
					return nil
				}
				return err
			}
			if line == "" || line[0] == '#' {
				continue
			}
			fields := strings.Split(line, ":")
			if len(fields) != 2 {
				panic("hoge")
			}
			for i, _ := range fields {
				fields[i] = strings.TrimSpace(fields[i])
			}
			if i < len(numFieldNames) {
				x, err := data.ToFloat(data.String(fields[1]))
				if err != nil {
					panic(err)
				}
				fv[numFieldNames[i]] = data.Float(x)
				i++
			} else {
				if fields[0] != "aspect" {
					panic(fields)
				}
				aspect := strings.Trim(fields[1], "\"")
				fv[aspect] = data.Float(1)
				break
			}
		}
		now := time.Now()
		w.Write(ctx, &core.Tuple{
			Data: data.Map{
				"feature_vector": fv,
			},
			Timestamp:     now,
			ProcTimestamp: now,
		})
	}

	return nil
}

func (*rentSource) Stop(*core.Context) error {
	return nil
}

func (r *rentSource) readLine() (string, error) {
	l, _, err := r.r.ReadLine()
	if err != nil {
		return "", err
	}
	return strings.Trim(string(l), "\n "), err
}

func createTrainingSource(*core.Context, *bql.IOParams, data.Map) (core.Source, error) {
	return new("rent-data.csv", true)
}

func createTestSource(*core.Context, *bql.IOParams, data.Map) (core.Source, error) {
	return new("myhome.yml", false)
}

func new(path string, training bool) (*rentSource, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	r := bufio.NewReader(f)
	return &rentSource{f, r, training}, nil
}

func init() {
	bql.MustRegisterGlobalSourceCreator("rent_training", bql.SourceCreatorFunc(createTrainingSource))
	bql.MustRegisterGlobalSourceCreator("rent_test", bql.SourceCreatorFunc(createTestSource))
}

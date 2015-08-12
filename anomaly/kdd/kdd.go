package kdd

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"os"
	"pfi/sensorbee/jubatus/internal/perline"
	"pfi/sensorbee/sensorbee/bql"
	"pfi/sensorbee/sensorbee/core"
	"pfi/sensorbee/sensorbee/data"
	"strconv"
	"strings"
)

type source struct {
	path string
}

func (s *source) Parse(line string, lineNo int) (data.Map, error) {
	line = strings.TrimSpace(line)
	fields := strings.Split(line, ",")
	if len(fields) == 0 {
		return nil, perline.Pass
	}
	if len(fields) != len(schema) {
		return nil, fmt.Errorf("invalid csv line at %s:%d", s.path, lineNo)
	}

	fv := data.Map{}
	for i := range schema {
		if schema[i].isString {
			if schema[i].name != "label" {
				// key names on Jubatus
				key := fmt.Sprintf("%s$%s@str#bin/bin", schema[i].name, fields[i])
				fv[key] = data.Float(1)
			}
		} else {
			x, err := strconv.ParseFloat(fields[i], 32)
			if err != nil {
				return nil, fmt.Errorf("invalid input at %s:%d", s.path, lineNo)
			}
			if x != 0 {
				// key names on Jubatus
				key := schema[i].name + "@num"
				fv[key] = data.Float(x)
			}
		}
	}
	return data.Map{
		"feature_vector": fv,
	}, nil
}

var schema = []struct {
	name     string
	isString bool
}{
	{name: "duration"},
	{name: "protocol_type", isString: true},
	{name: "service", isString: true},
	{name: "flag", isString: true},
	{name: "src_bytes"},
	{name: "dst_bytes"},
	{name: "land", isString: true},
	{name: "wrong_fragment"},
	{name: "urgent"},
	{name: "hot"},
	{name: "num_failed_logins"},
	{name: "logged_in", isString: true},
	{name: "num_compromised"},
	{name: "root_shell"},
	{name: "su_attempted"},
	{name: "num_root"},
	{name: "num_file_creations"},
	{name: "num_shells"},
	{name: "num_access_files"},
	{name: "num_outbound_cmds"},
	{name: "is_host_login", isString: true},
	{name: "is_guest_login", isString: true},
	{name: "count"},
	{name: "srv_count"},
	{name: "serror_rate"},
	{name: "srv_serror_rate"},
	{name: "rerror_rate"},
	{name: "srv_rerror_rate"},
	{name: "same_srv_rate"},
	{name: "diff_srv_rate"},
	{name: "srv_diff_host_rate"},
	{name: "dst_host_count"},
	{name: "dst_host_srv_count"},
	{name: "dst_host_same_srv_rate"},
	{name: "dst_host_diff_srv_rate"},
	{name: "dst_host_same_src_port_rate"},
	{name: "dst_host_srv_diff_host_rate"},
	{name: "dst_host_serror_rate"},
	{name: "dst_host_srv_serror_rate"},
	{name: "dst_host_rerror_rate"},
	{name: "dst_host_srv_rerror_rate"},
	{name: "label", isString: true},
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
	bql.MustRegisterGlobalSourceCreator("kdd_source", bql.SourceCreatorFunc(createSource))
}

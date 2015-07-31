package perline

import (
	"bufio"
	"errors"
	"io"
	"pfi/sensorbee/sensorbee/core"
	"pfi/sensorbee/sensorbee/data"
	"strings"
	"time"
)

type Source struct {
	lp          LineParser
	bufio       BufioReaderFactory
	firstLineNo int
}

type LineParser interface {
	Parse(line string, lineNo int) (data.Map, error)
}
type BufioReaderFactory interface {
	Reader() (*bufio.Reader, io.Closer, error)
}

func NewSource(lp LineParser, bufio BufioReaderFactory, firstLineNo int) *Source {
	return &Source{
		lp:          lp,
		bufio:       bufio,
		firstLineNo: firstLineNo,
	}
}

func (s *Source) GenerateStream(ctx *core.Context, w core.Writer) error {
	r, c, err := s.bufio.Reader()
	if err != nil {
		return err
	}
	defer c.Close()

	for lineNo := s.firstLineNo; ; lineNo++ {
		line, err := r.ReadString('\n')
		line = strings.TrimSpace(line)
		if err != nil {
			if err != io.EOF {
				return err
			}
			if len(line) == 0 {
				return nil
			}
		}
		data, err := s.lp.Parse(line, lineNo)
		if err != nil {
			if err == Pass {
				continue
			}
			return err
		}
		now := time.Now()
		err = w.Write(ctx, &core.Tuple{
			Data:          data,
			Timestamp:     now,
			ProcTimestamp: now,
		})
		if err != nil {
			return err
		}
	}
}

func (s *Source) Stop(*core.Context) error {
	return nil
}

var Pass = errors.New("pass")

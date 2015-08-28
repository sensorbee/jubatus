package nested

import (
	"fmt"
	"pfi/sensorbee/sensorbee/data"
)

type Appender func(string, float32)

func Flatten(v data.Map, ap Appender) error {
	return flattenImpl("", v, ap)
}

func flattenImpl(keyPrefix string, v data.Map, ap Appender) error {
	for f, val := range v {
		if m, err := data.AsMap(val); err == nil {
			err := flattenImpl(fmt.Sprint(keyPrefix, f, "\x00"), m, ap)
			if err != nil {
				return err
			}
		} else {
			xx, err := data.ToFloat(val)
			if err != nil {
				// TODO: return better error
				return err
			}
			x := float32(xx)
			ap(keyPrefix+f, x)
		}
	}
	return nil
}

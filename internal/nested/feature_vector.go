package nested

import (
	"fmt"
	"pfi/sensorbee/sensorbee/data"
)

type Appender func(string, float32)

func Flatten(v data.Map, ap Appender) error {
	return flattenImpl("", v, ap)
}

func flattenImpl(keyPrefix string, v data.Value, ap Appender) error {
	switch v.Type() {
	case data.TypeArray:
		keyPrefix += "\x00"
		a, _ := data.AsArray(v)
		for i, v := range a {
			err := flattenImpl(fmt.Sprintf(keyPrefix, i), v, ap)
			if err != nil {
				return err
			}
		}

	case data.TypeMap:
		if keyPrefix != "" {
			keyPrefix += "\x00"
		}

		m, _ := data.AsMap(v)
		for f, v := range m {
			err := flattenImpl(keyPrefix+f, v, ap)
			if err != nil {
				return err
			}
		}

	default:
		xx, err := data.ToFloat(v)
		if err != nil {
			// TODO: return better error
			return err
		}
		x := float32(xx)
		ap(keyPrefix, x)
	}
	return nil
}

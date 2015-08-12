package pluginutil

import (
	"fmt"
	"pfi/sensorbee/sensorbee/data"
)

func ExtractParamAsStringWithDefault(params data.Map, key, def string) (string, error) {
	v, ok := params[key]
	if !ok {
		return def, nil
	}

	s, err := data.AsString(v)
	if err != nil {
		return "", fmt.Errorf("%s parameter is not a string: %v", key, err)
	}
	return s, nil
}

func ExtractParamAsString(params data.Map, key string) (string, error) {
	v, ok := params[key]
	if !ok {
		return "", fmt.Errorf("%s parameter is missing", key)
	}
	s, err := data.AsString(v)
	if err != nil {
		return "", fmt.Errorf("%s parameter is not a string: %v", err)
	}
	return s, nil
}

func ExtractParamAsInt(params data.Map, key string) (int64, error) {
	v, ok := params[key]
	if !ok {
		return 0, fmt.Errorf("%s parameter is missing", key)
	}
	x, err := data.AsInt(v)
	if err != nil {
		return 0, fmt.Errorf("%s parameter is not an integer: %v", key, err)
	}
	return x, nil
}

func ExtractParamAndConvertToFloat(params data.Map, key string) (float64, error) {
	v, ok := params[key]
	if !ok {
		return 0, fmt.Errorf("%s parameter is missing", key)
	}
	x, err := data.ToFloat(v)
	if err != nil {
		return 0, fmt.Errorf("%s parameter cannot be converted to float: %v", key, err)
	}
	return x, nil
}

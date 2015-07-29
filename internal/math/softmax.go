package math

import (
	"math"
	"pfi/sensorbee/sensorbee/data"
	"sort"
)

func Softmax(v data.Map) (data.Map, error) {
	ret := make(data.Map)

	if len(v) == 0 {
		return ret, nil
	}

	// copy values to an array to sort in logSumExp().
	values, err := mapToValues(v)
	if err != nil {
		return nil, err
	}

	lse := logSumExp(values)
	for k, x := range v {
		val, err := data.AsFloat(x)
		if err != nil {
			return nil, err
		}
		ret[k] = data.Float(math.Exp(val - lse))
	}
	return ret, nil
}

func mapToValues(v data.Map) ([]float64, error) {
	values := make([]float64, 0, len(v))
	for _, x := range v {
		val, err := data.AsFloat(x)
		if err != nil {
			return nil, err
		}
		values = append(values, val)
	}
	return values, nil
}

// logSumExp calculates logsumexp. When calling this function with an empty
// slice, the behavior is undefined.
func logSumExp(v []float64) float64 {
	if len(v) == 0 {
		return math.NaN()
	}

	// TODO: consider better calculation order to reduce floating-point error.
	sort.Float64s(v)
	x := v[0]
	for i := 1; i < len(v); i++ {
		y := v[i]
		if x < y {
			x, y = y, x
		}
		x += math.Log(1 + math.Exp(y-x))
	}
	return x
}

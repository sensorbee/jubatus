package regression

import (
	"errors"
	"math"
	"pfi/sensorbee/sensorbee/data"
	"sync"
)

type PassiveAggressive struct {
	model
	sum   float32
	sqSum float32
	count uint64
	m     sync.RWMutex

	regWeight   float32
	sensitivity float32
}

func NewPassiveAggressive(regWeight float32, sensitivity float32) (*PassiveAggressive, error) {
	if regWeight <= 0 {
		return nil, errors.New("TODO")
	}
	if sensitivity < 0 {
		return nil, errors.New("TODO")
	}
	return &PassiveAggressive{
		model:       make(model),
		regWeight:   regWeight,
		sensitivity: sensitivity,
	}, nil
}

func (pa *PassiveAggressive) Train(v FeatureVector, value float32) error {
	fv, err := v.toInternal()
	if err != nil {
		return err
	}

	pa.m.Lock()
	defer pa.m.Unlock()

	pa.sum += value
	pa.sqSum += value * value
	pa.count++

	avg := pa.sum / float32(pa.count)
	stdDev := sqrt(pa.sqSum/float32(pa.count) - avg*avg)

	predict := pa.estimate(fv)
	error := value - predict
	loss := abs(error) - pa.sensitivity*stdDev

	if loss <= 0 {
		return nil
	}

	C := pa.regWeight
	coeff := sign(error) * min(C, loss) / fv.squaredNorm()
	pa.update(fv, coeff)
	return nil
}

func (pa *PassiveAggressive) Estimate(v FeatureVector) (float32, error) {
	fv, err := v.toInternal()
	if err != nil {
		return 0, err
	}

	pa.m.RLock()
	defer pa.m.RUnlock()

	return pa.estimate(fv), nil
}

func (pa *PassiveAggressive) Clear() {
	pa.m.Lock()
	defer pa.m.Unlock()

	pa.model = make(model)
	pa.sum = 0
	pa.sqSum = 0
	pa.count = 0
}

func (pa *PassiveAggressive) RegWeight() float32 {
	return pa.regWeight
}

func (pa *PassiveAggressive) Sensitivity() float32 {
	return pa.sensitivity
}

func (pa *PassiveAggressive) estimate(v fVector) float32 {
	var ret float32
	for i := range v {
		dim := v[i].dim
		x := v[i].value

		ret += x * pa.model[dim]
	}
	return ret
}

func (pa *PassiveAggressive) update(v fVector, coeff float32) {
	for i := range v {
		dim := v[i].dim
		x := v[i].value

		pa.model[dim] += coeff * x
	}
}

type dim string

type model map[dim]float32

type FeatureVector data.Map

func (v FeatureVector) toInternal() (fVector, error) {
	ret := make(fVector, 0, len(v))
	for k, v := range v {
		x, err := data.ToFloat(v)
		if err != nil {
			return nil, err
		}
		ret = append(ret, fElement{dim(k), float32(x)})
	}
	return ret, nil
}

type fElement struct {
	dim
	value float32
}
type fVector []fElement

func (v fVector) squaredNorm() float32 {
	var norm2 float32
	for i := 0; i < len(v); i++ {
		val := v[i].value
		norm2 += val * val
	}
	return norm2
}

func abs(x float32) float32 {
	return float32(math.Abs(float64(x)))
}

func min(x float32, y float32) float32 {
	return float32(math.Min(float64(x), float64(y)))
}

func sqrt(x float32) float32 {
	return float32(math.Sqrt(float64(x)))
}

func sign(x float32) float32 {
	return float32(math.Copysign(1, float64(x)))
}

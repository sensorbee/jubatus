package regression

import (
	"errors"
	"fmt"
	"github.com/sensorbee/jubatus/internal/nested"
	"github.com/ugorji/go/codec"
	"gopkg.in/sensorbee/sensorbee.v0/data"
	"io"
	"math"
	"sync"
)

// PassiveAggressive holds a model for regression.
type PassiveAggressive struct {
	model model
	sum   float32
	sqSum float32
	count uint64
	m     sync.RWMutex

	regWeight   float32
	sensitivity float32
}

// NewPassiveAggressive creates a PassiveAggressive model. regWeight must be greater than zero.
// sensitivity must not be less than zero.
func NewPassiveAggressive(regWeight float32, sensitivity float32) (*PassiveAggressive, error) {
	if regWeight <= 0 {
		return nil, errors.New("regularization weight must be larger than zero")
	}
	if sensitivity < 0 {
		return nil, errors.New("sensitivity must not be less than zero")
	}
	return &PassiveAggressive{
		model:       make(model),
		regWeight:   regWeight,
		sensitivity: sensitivity,
	}, nil
}

// Train trains a model with a feature vector and a value.
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

	// zero vector generates inf or nan.
	if fv.squaredNorm() < 1e-12 {
		return nil
	}

	C := pa.regWeight
	coeff := sign(error) * min(C, loss) / fv.squaredNorm()
	pa.update(fv, coeff)
	return nil
}

// Estimate estimates a value from a model and a feature vector.
func (pa *PassiveAggressive) Estimate(v FeatureVector) (float32, error) {
	fv, err := v.toInternal()
	if err != nil {
		return 0, err
	}

	pa.m.RLock()
	defer pa.m.RUnlock()

	return pa.estimate(fv), nil
}

// Clear clears a model.
func (pa *PassiveAggressive) Clear() {
	pa.m.Lock()
	defer pa.m.Unlock()

	pa.model = make(model)
	pa.sum = 0
	pa.sqSum = 0
	pa.count = 0
}

const (
	paForwatVersion = 1
)

type paMsgpack struct {
	_struct struct{} `codec:",toarray"`

	Model model
	Sum   float32
	SqSum float32
	Count uint64

	RegWeight   float32
	Sensitivity float32
}

// Save saves the current state of PassiveAggressive.
func (pa *PassiveAggressive) Save(w io.Writer) error {
	pa.m.RLock()
	defer pa.m.RUnlock()

	if _, err := w.Write([]byte{paForwatVersion}); err != nil {
		return err
	}

	enc := codec.NewEncoder(w, regressionMsgpackHandle)
	err := enc.Encode(&paMsgpack{
		Model:       pa.model,
		Sum:         pa.sum,
		SqSum:       pa.sqSum,
		Count:       pa.count,
		RegWeight:   pa.regWeight,
		Sensitivity: pa.sensitivity,
	})
	return err
}

// LoadPassiveAggressive loads PassiveAggressive from the saved data.
func LoadPassiveAggressive(r io.Reader) (*PassiveAggressive, error) {
	formatVersion := make([]byte, 1)
	if _, err := r.Read(formatVersion); err != nil {
		return nil, err
	}

	switch formatVersion[0] {
	case 1:
		return loadPassiveAggressiveFormatV1(r)
	default:
		return nil, fmt.Errorf("unsupported format version of PassiveAggressive container: %v", formatVersion[0])
	}
}

func loadPassiveAggressiveFormatV1(r io.Reader) (*PassiveAggressive, error) {
	m := paMsgpack{}
	dec := codec.NewDecoder(r, regressionMsgpackHandle)
	if err := dec.Decode(&m); err != nil {
		return nil, err
	}

	return &PassiveAggressive{
		model: m.Model,
		sum:   m.Sum,
		sqSum: m.SqSum,
		count: m.Count,

		regWeight:   m.RegWeight,
		sensitivity: m.Sensitivity,
	}, nil
}

// RegWeight returns regularization weight.
func (pa *PassiveAggressive) RegWeight() float32 {
	return pa.regWeight
}

// Sensitivity returns sensitivity.
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

// FeatureVector is a type for feature vectors.
type FeatureVector data.Map

func (v FeatureVector) toInternal() (fVector, error) {
	ret := make(fVector, 0, len(v))
	err := nested.Flatten(data.Map(v), func(key string, value float32) {
		ret = append(ret, fElement{dim: dim(key), value: value})
	})
	if err != nil {
		return nil, err
	}
	return ret, nil
}

type fElement struct {
	dim   dim
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

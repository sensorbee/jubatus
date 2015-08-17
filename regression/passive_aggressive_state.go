package regression

import (
	"errors"
	"fmt"
	"pfi/sensorbee/jubatus/internal/pluginutil"
	"pfi/sensorbee/sensorbee/core"
	"pfi/sensorbee/sensorbee/data"
)

type PassiveAggressiveState struct {
	pa                 *PassiveAggressive
	valueField         string
	featureVectorField string
}

func NewPassiveAggressiveState(ctx *core.Context, params data.Map) (core.SharedState, error) {
	value, err := pluginutil.ExtractParamAsStringWithDefault(params, "value_field", "value")
	if err != nil {
		return nil, err
	}
	fv, err := pluginutil.ExtractParamAsStringWithDefault(params, "feature_vector_field", "feature_vector")
	if err != nil {
		return nil, err
	}

	rw, err := pluginutil.ExtractParamAndConvertToFloat(params, "regularization_weight")
	if err != nil {
		return nil, err
	}
	if rw <= 0 {
		return nil, errors.New("regularization_weight parameter must be greater than zero")
	}

	sen, err := pluginutil.ExtractParamAndConvertToFloat(params, "sensitivity")
	if err != nil {
		return nil, err
	}
	if sen < 0 {
		return nil, errors.New("sensitivity parameter must be not less than zero")
	}

	pa, err := NewPassiveAggressive(float32(rw), float32(sen))
	if err != nil {
		return nil, err
	}

	return &PassiveAggressiveState{
		pa:                 pa,
		valueField:         value,
		featureVectorField: fv,
	}, nil
}

func (*PassiveAggressiveState) Terminate(ctx *core.Context) error {
	return nil
}

func (pa *PassiveAggressiveState) Write(ctx *core.Context, t *core.Tuple) error {
	vval, ok := t.Data[pa.valueField]
	if !ok {
		return fmt.Errorf("%s field is missing", pa.valueField)
	}
	val64, err := data.ToFloat(vval)
	if err != nil {
		return fmt.Errorf("%s cannot be converted to float: %v", pa.valueField, err)
	}
	val := float32(val64)

	vfv, ok := t.Data[pa.featureVectorField]
	if !ok {
		return fmt.Errorf("%s field is missing", pa.featureVectorField)
	}
	fv, err := data.AsMap(vfv)
	if err != nil {
		return fmt.Errorf("%s value is not a map: %v", pa.featureVectorField, err)
	}

	err = pa.pa.Train(FeatureVector(fv), val)
	return err
}

func PassiveAggressiveEstimate(ctx *core.Context, stateName string, featureVector data.Map) (float32, error) {
	s, err := lookupPassiveAggressiveState(ctx, stateName)
	if err != nil {
		return 0, err
	}

	return s.pa.Estimate(FeatureVector(featureVector))
}

func lookupPassiveAggressiveState(ctx *core.Context, stateName string) (*PassiveAggressiveState, error) {
	st, err := ctx.SharedStates.Get(stateName)
	if err != nil {
		return nil, err
	}

	if s, ok := st.(*PassiveAggressiveState); ok {
		return s, nil
	}
	return nil, fmt.Errorf("state '%v' cannot be converted to paState", stateName)
}

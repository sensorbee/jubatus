package plugin

import (
	"errors"
	"fmt"
	"pfi/sensorbee/jubatus/internal/pluginutil"
	"pfi/sensorbee/jubatus/regression"
	"pfi/sensorbee/sensorbee/bql/udf"
	"pfi/sensorbee/sensorbee/core"
	"pfi/sensorbee/sensorbee/data"
)

func init() {
	if err := udf.RegisterGlobalUDSCreator("jubaregression_pa", udf.UDSCreatorFunc(newPAState)); err != nil {
		panic(err)
	}
	if err := udf.RegisterGlobalUDF("jubaregression_estimate", udf.MustConvertGeneric(paEstimate)); err != nil {
		panic(err)
	}
}

type paState struct {
	pa                 *regression.PassiveAggressive
	valueField         string
	featureVectorField string
}

func newPAState(ctx *core.Context, params data.Map) (core.SharedState, error) {
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

	pa, err := regression.NewPassiveAggressive(float32(rw), float32(sen))
	if err != nil {
		return nil, err
	}

	return &paState{
		pa:                 pa,
		valueField:         value,
		featureVectorField: fv,
	}, nil
}

func (*paState) Terminate(ctx *core.Context) error {
	return nil
}

func (pa *paState) Write(ctx *core.Context, t *core.Tuple) error {
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

	err = pa.pa.Train(regression.FeatureVector(fv), val)
	return err
}

func paEstimate(ctx *core.Context, stateName string, featureVector data.Map) (float32, error) {
	s, err := lookupPAState(ctx, stateName)
	if err != nil {
		return 0, err
	}

	return s.pa.Estimate(regression.FeatureVector(featureVector))
}

func lookupPAState(ctx *core.Context, stateName string) (*paState, error) {
	st, err := ctx.SharedStates.Get(stateName)
	if err != nil {
		return nil, err
	}

	if s, ok := st.(*paState); ok {
		return s, nil
	}
	return nil, fmt.Errorf("state '%v' cannot be converted to paState", stateName)
}

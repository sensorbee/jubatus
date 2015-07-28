package plugin

import (
	"errors"
	"fmt"
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
	pa *regression.PassiveAggressive
}

func newPAState(ctx *core.Context, params data.Map) (core.SharedState, error) {
	rw, err := extractFloat32Parameter(params, "regularization_weight")
	if err != nil {
		return nil, err
	}
	if rw <= 0 {
		return nil, errors.New("regularization_weight parameter must be greater than zero")
	}

	sen, err := extractFloat32Parameter(params, "sensitivity")
	if err != nil {
		return nil, err
	}
	if sen < 0 {
		return nil, errors.New("sensitivity parameter must be not less than zero")
	}

	pa, err := regression.NewPassiveAggressive(rw, sen)
	if err != nil {
		return nil, err
	}

	return &paState{
		pa: pa,
	}, nil
}

func (*paState) Terminate(ctx *core.Context) error {
	return nil
}

func (pa *paState) Write(ctx *core.Context, t *core.Tuple) error {
	vval, ok := t.Data["value"]
	if !ok {
		return errors.New("value field is missing")
	}
	val64, err := data.ToFloat(vval)
	if err != nil {
		return fmt.Errorf("value cannot be converted to float: %v", err)
	}
	val := float32(val64)

	vfv, ok := t.Data["feature_vector"]
	if !ok {
		return errors.New("feature_vector field is missing")
	}
	fv, err := data.AsMap(vfv)
	if err != nil {
		return fmt.Errorf("feature_vector value is not a map: %v", err)
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

func extractFloat32Parameter(m data.Map, name string) (float32, error) {
	v, ok := m[name]
	if !ok {
		return 0, fmt.Errorf("%s parameter is missing", name)
	}
	x, err := data.ToFloat(v)
	if err != nil {
		return 0, fmt.Errorf("%s parameter cannot be converted to float: %v", err)
	}

	return float32(x), nil
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

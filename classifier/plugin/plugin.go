package plugin

import (
	"errors"
	"fmt"
	"pfi/sensorbee/jubatus/classifier"
	"pfi/sensorbee/sensorbee/bql/udf"
	"pfi/sensorbee/sensorbee/core"
	"pfi/sensorbee/sensorbee/data"
)

func init() {
	if err := udf.RegisterGlobalUDSCreator("jubaclassifier_arow", udf.UDSCreatorFunc(newAROWState)); err != nil {
		panic(err)
	}
	if err := udf.RegisterGlobalUDF("jubaclassify", udf.MustConvertGeneric(arowClassify)); err != nil {
		panic(err)
	}
}

type arowState struct {
	arow               *classifier.AROW
	labelField         string
	featureVectorField string
}

func newAROWState(ctx *core.Context, params data.Map) (core.SharedState, error) {
	label, err := extractParamAsStringWithDefault(params, "label_field", "label")
	if err != nil {
		return nil, err
	}
	fv, err := extractParamAsStringWithDefault(params, "feature_vector_field", "feature_vector")
	if err != nil {
		return nil, err
	}

	v, ok := params["regularization_weight"]
	if !ok {
		return nil, errors.New("regularization_weight parameter is missing")
	}

	rw, err := data.ToFloat(v)
	if err != nil {
		return nil, fmt.Errorf("regularization_weight parameter cannot be converted to a float: %v", err)
	}
	if rw <= 0 {
		return nil, errors.New("regularization_weight parameter must be greater than zero")
	}

	a, err := classifier.NewAROW(float32(rw))
	if err != nil {
		return nil, fmt.Errorf("failed to initialize AROW: %v", err)
	}

	return &arowState{
		arow:               a,
		labelField:         label,
		featureVectorField: fv,
	}, nil
}

func (*arowState) Terminate(ctx *core.Context) error {
	return nil
}

func (a *arowState) Write(ctx *core.Context, t *core.Tuple) error {
	vlabel, ok := t.Data[a.labelField]
	if !ok {
		return fmt.Errorf("%s field is missing", a.labelField)
	}
	label, err := data.AsString(vlabel)
	if err != nil {
		return fmt.Errorf("%s value is not a string: %v", a.labelField, err)
	}

	vfv, ok := t.Data[a.featureVectorField]
	if !ok {
		return fmt.Errorf("%s field is missing", a.featureVectorField)
	}
	fv, err := data.AsMap(vfv)
	if err != nil {
		return fmt.Errorf("%s value is not a map: %v", a.labelField, err)
	}

	err = a.train(fv, label)
	return err
}

func (a *arowState) train(fv data.Map, l string) error {
	return a.arow.Train(classifier.FeatureVector(fv), classifier.Label(l))
}

func arowClassify(ctx *core.Context, stateName string, featureVector data.Map) (data.Map, error) {
	s, err := lookupAROWState(ctx, stateName)
	if err != nil {
		return nil, err
	}

	scores, err := s.arow.Classify(classifier.FeatureVector(featureVector))
	return data.Map(scores), err
}

func lookupAROWState(ctx *core.Context, stateName string) (*arowState, error) {
	st, err := ctx.SharedStates.Get(stateName)
	if err != nil {
		return nil, err
	}

	if s, ok := st.(*arowState); ok {
		return s, nil
	}
	return nil, fmt.Errorf("state '%v' cannot be converted to arowState", stateName)
}

func extractParamAsStringWithDefault(params data.Map, key, def string) (string, error) {
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

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
	if err := udf.RegisterGlobalUDF("jubaclassifier_arow_train", udf.MustConvertGeneric(arowTrain)); err != nil {
		panic(err)
	}
	if err := udf.RegisterGlobalUDF("jubaclassifier_arow_classify", udf.MustConvertGeneric(arowClassify)); err != nil {
		panic(err)
	}
}

type arowState struct {
	*classifier.AROW
}

func newAROWState(ctx *core.Context, params data.Map) (core.SharedState, error) {
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
		AROW: a,
	}, nil
}

func (*arowState) Terminate(ctx *core.Context) error {
	return nil
}

func (a *arowState) Write(ctx *core.Context, t *core.Tuple) error {
	vlabel, ok := t.Data["label"]
	if !ok {
		return errors.New("label field is missing")
	}
	label, err := data.ToString(vlabel)
	if err != nil {
		return fmt.Errorf("label value cannot be converted to a string: %v", err)
	}

	vfv, ok := t.Data["feature_vector"]
	if !ok {
		return errors.New("feature_vector field is missing")
	}
	fv, err := data.AsMap(vfv)
	if err != nil {
		return fmt.Errorf("feature_vector value is not a map: %v", err)
	}

	err = a.train(fv, label)
	return err
}

func (a *arowState) train(fv data.Map, l string) error {
	return a.AROW.Train(classifier.FeatureVector(fv), classifier.Label(l))
}

func arowTrain(ctx *core.Context, stateName string, featureVector data.Map, label string) (string, error) {
	s, err := lookupAROWState(ctx, stateName)
	if err != nil {
		return "", err
	}

	err = s.train(featureVector, label)
	if err != nil {
		return "", err
	}
	return label, nil
}

func arowClassify(ctx *core.Context, stateName string, featureVector data.Map) (data.Map, error) {
	s, err := lookupAROWState(ctx, stateName)
	if err != nil {
		return nil, err
	}

	scores, err := s.AROW.Classify(classifier.FeatureVector(featureVector))
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

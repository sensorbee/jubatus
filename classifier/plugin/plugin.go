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
	mfv, err := data.AsMap(vfv)
	if err != nil {
		return fmt.Errorf("feature_vector value is not a map: %v", err)
	}
	fv, err := mapToFeatureVector(mfv)
	if err != nil {
		return err
	}

	err = a.AROW.Train(fv, classifier.Label(label))
	return err
}

func arowTrain(ctx *core.Context, stateName string, featureVector data.Map, label string) (string, error) {
	s, err := lookupAROWState(ctx, stateName)
	if err != nil {
		return "", err
	}

	fv, err := mapToFeatureVector(featureVector)
	if err != nil {
		return "", err
	}

	err = s.AROW.Train(fv, classifier.Label(label))
	return label, err
}

func arowClassify(ctx *core.Context, stateName string, featureVector data.Map) (data.Map, error) {
	s, err := lookupAROWState(ctx, stateName)
	if err != nil {
		return nil, err
	}

	fv, err := mapToFeatureVector(featureVector)
	if err != nil {
		return nil, err
	}

	scores := s.AROW.Classify(fv)
	ret := make(data.Map)
	for i, _ := range scores {
		lscore := &scores[i]
		ret[string(lscore.Label)] = data.Float(lscore.Score)
	}
	return ret, nil
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

func mapToFeatureVector(m data.Map) (classifier.FeatureVector, error) {
	fv := make(classifier.FeatureVector, 0, len(m))
	for k, v := range m {
		x, err := valueToFloat32(v)
		if err != nil {
			return nil, err
		}
		fv = append(fv, classifier.FeatureElement{k, x})
	}
	return fv, nil
}

func valueToFloat32(v data.Value) (float32, error) {
	f64, err := data.ToFloat(v)
	if err != nil {
		return 0, err
	}
	return float32(f64), nil
}

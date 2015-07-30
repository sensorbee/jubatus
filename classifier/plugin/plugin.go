package plugin

import (
	"errors"
	"fmt"
	stdMath "math"
	"pfi/sensorbee/jubatus/classifier"
	"pfi/sensorbee/jubatus/internal/math"
	"pfi/sensorbee/jubatus/internal/pluginutil"
	"pfi/sensorbee/sensorbee/bql/udf"
	"pfi/sensorbee/sensorbee/core"
	"pfi/sensorbee/sensorbee/data"
)

func init() {
	if err := udf.RegisterGlobalUDSCreator("jubaclassifier_arow", udf.UDSCreatorFunc(newAROWState)); err != nil {
		panic(err)
	}

	// The name jubaclassify is not only for AROW, but for all classifier algorithms.
	// We have implemented only AROW, so we use the name for arowClassify for now.
	// When we have to implement another classification algorithm, generalize jubaclassify
	// to other algorithms. For example, define classifier.Classifier and adjust all algorithms to it.
	if err := udf.RegisterGlobalUDF("jubaclassify", udf.MustConvertGeneric(arowClassify)); err != nil {
		panic(err)
	}

	// TODO: consider to rename
	if err := udf.RegisterGlobalUDF("juba_classified_label", udf.MustConvertGeneric(classifiedLabel)); err != nil {
		panic(err)
	}

	if err := udf.RegisterGlobalUDF("juba_classified_score", udf.MustConvertGeneric(classifiedScore)); err != nil {
		panic(err)
	}

	if err := udf.RegisterGlobalUDF("juba_softmax", udf.MustConvertGeneric(math.Softmax)); err != nil {
		panic(err)
	}
}

type arowState struct {
	arow               *classifier.AROW
	labelField         string
	featureVectorField string
}

func newAROWState(ctx *core.Context, params data.Map) (core.SharedState, error) {
	label, err := pluginutil.ExtractParamAsStringWithDefault(params, "label_field", "label")
	if err != nil {
		return nil, err
	}
	fv, err := pluginutil.ExtractParamAsStringWithDefault(params, "feature_vector_field", "feature_vector")
	if err != nil {
		return nil, err
	}
	rw, err := pluginutil.ExtractParamAndConvertToFloat(params, "regularization_weight")
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

func classifiedLabel(ctx *core.Context, scores data.Map) (string, error) {
	if len(scores) == 0 {
		return "", errors.New("attempt to get a label from an empty map")
	}

	// classifier.LScores.Max() cannot be used here because
	// scores is passed by a user. classifier.LScores.Max()
	// expects all values are float.
	l, _, err := maxLabelScore(scores)
	if err != nil {
		return "", err
	}
	return l, nil
}

func classifiedScore(ctx *core.Context, scores data.Map) (float64, error) {
	if len(scores) == 0 {
		return 0, errors.New("attempt to get a score from an empty map")
	}

	_, s, err := maxLabelScore(scores)
	if err != nil {
		return 0, err
	}
	return s, nil
}

// maxLabelScore returns the max score and its label in a data.Map.
// This function are same as classifier.LScores.Max() except error checking.
func maxLabelScore(scores data.Map) (label string, score float64, err error) {
	if len(scores) == 0 {
		err = errors.New("attempt to find a max score from an empty map")
		return "", 0, err
	}

	score = minusInf
	for l, s := range scores {
		sc, err := data.AsFloat(s)
		if err != nil {
			err = fmt.Errorf("score for %s is not a float: %v", l, err)
			return "", 0, err
		}
		if sc > score {
			label = l
			score = sc
		}
	}

	return label, score, nil
}

var minusInf = stdMath.Inf(-1)

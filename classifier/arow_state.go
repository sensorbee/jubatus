package classifier

import (
	"errors"
	"fmt"
	"github.com/ugorji/go/codec"
	"io"
	stdMath "math"
	"pfi/sensorbee/jubatus/internal/pluginutil"
	"pfi/sensorbee/sensorbee/bql/udf"
	"pfi/sensorbee/sensorbee/core"
	"pfi/sensorbee/sensorbee/data"
	"reflect"
)

// classfierMsgpack has information of the saved file.
type classifierMsgpack struct {
	_struct       struct{} `codec:",toarray"`
	FormatVersion uint8
	Algorithm     string
}

// AROWState is a state which support AROW classification algorithm.
type AROWState struct {
	arow               *AROW
	labelField         string
	featureVectorField string
}

var _ core.SavableSharedState = &AROWState{}

type arowStateMsgpack struct {
	_struct            struct{} `codec:",toarray"`
	LabelField         string
	FeatureVectorField string
}

// AROWStateCreator is used by BQL to create or load AROWState as a UDS.
type AROWStateCreator struct {
}

var _ udf.UDSLoader = &AROWStateCreator{}

// CreateState creates a new state for AROW classifier.
func (c *AROWStateCreator) CreateState(ctx *core.Context, params data.Map) (core.SharedState, error) {
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

	a, err := NewAROW(float32(rw))
	if err != nil {
		return nil, fmt.Errorf("failed to initialize AROW: %v", err)
	}

	return &AROWState{
		arow:               a,
		labelField:         label,
		featureVectorField: fv,
	}, nil
}

var (
	classifierMsgpackHandle = &codec.MsgpackHandle{
		RawToString: true,
	}
)

func init() {
	classifierMsgpackHandle.MapType = reflect.TypeOf(map[string]interface{}{})
}

// LoadState loads a new state for AROW classifier.
func (c *AROWStateCreator) LoadState(ctx *core.Context, r io.Reader, params data.Map) (core.SharedState, error) {
	var d classifierMsgpack
	dec := codec.NewDecoder(r, classifierMsgpackHandle)
	if err := dec.Decode(&d); err != nil {
		return nil, err
	}
	if d.Algorithm != "arow" {
		return nil, fmt.Errorf("unsupported classification algorithm: %v", d.Algorithm)
	}

	switch d.FormatVersion {
	case 1:
		return loadAROWStateFormatV1(ctx, r)
	default:
		return nil, fmt.Errorf("unsupported format version of AROWState container: %v", d.FormatVersion)
	}
}

func loadAROWStateFormatV1(ctx *core.Context, r io.Reader) (core.SharedState, error) {
	// This is the current format and no data type conversion is required.
	s := &AROWState{}

	var d arowStateMsgpack
	dec := codec.NewDecoder(r, classifierMsgpackHandle)
	if err := dec.Decode(&d); err != nil {
		return nil, err
	}
	s.labelField = d.LabelField
	s.featureVectorField = d.FeatureVectorField

	arow, err := LoadAROW(r)
	if err != nil {
		return nil, err
	}
	s.arow = arow
	return s, nil
}

// Terminate terminates the state.
func (*AROWState) Terminate(ctx *core.Context) error {
	return nil
}

// Write trains the machine learning model the state has with a given tuple.
func (a *AROWState) Write(ctx *core.Context, t *core.Tuple) error {
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

func (a *AROWState) train(fv data.Map, l string) error {
	return a.arow.Train(FeatureVector(fv), Label(l))
}

const (
	classifierFormatVersion uint8 = 1
)

// Save is provided as a part of core.SavableSharedState.
func (a *AROWState) Save(ctx *core.Context, w io.Writer, params data.Map) error {
	// This is the format version of the root container and doesn't related to
	// how each algorithm is saved.
	enc := codec.NewEncoder(w, classifierMsgpackHandle)
	if err := enc.Encode(&classifierMsgpack{
		FormatVersion: classifierFormatVersion,
		Algorithm:     "arow",
	}); err != nil {
		return err
	}

	if err := enc.Encode(&arowStateMsgpack{
		LabelField:         a.labelField,
		FeatureVectorField: a.featureVectorField,
	}); err != nil {
		return err
	}
	return a.arow.Save(w)
}

// AROWClassify classifies the input using the given model having stateName.
func AROWClassify(ctx *core.Context, stateName string, featureVector data.Map) (data.Map, error) {
	s, err := lookupAROWState(ctx, stateName)
	if err != nil {
		return nil, err
	}

	scores, err := s.arow.Classify(FeatureVector(featureVector))
	return data.Map(scores), err
}

func lookupAROWState(ctx *core.Context, stateName string) (*AROWState, error) {
	st, err := ctx.SharedStates.Get(stateName)
	if err != nil {
		return nil, err
	}

	if s, ok := st.(*AROWState); ok {
		return s, nil
	}
	return nil, fmt.Errorf("state '%v' isn't an AROWState", stateName)
}

// ClassifiedLabel returns the label having the highest score in a
// classification result.
func ClassifiedLabel(scores data.Map) (string, error) {
	if len(scores) == 0 {
		return "", errors.New("attempt to get a label from an empty map")
	}

	// LScores.Max() cannot be used here because scores is passed by a user.
	// LScores.Max() expects all values are float.
	l, _, err := maxLabelScore(scores)
	if err != nil {
		return "", err
	}
	return l, nil
}

// ClassifiedScore returns the highest score in a classification result.
func ClassifiedScore(scores data.Map) (float64, error) {
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
// This function are same as LScores.Max() except error checking.
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

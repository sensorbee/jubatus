package regression

import (
	"errors"
	"fmt"
	"github.com/sensorbee/jubatus/internal/pluginutil"
	"github.com/ugorji/go/codec"
	"gopkg.in/sensorbee/sensorbee.v0/bql/udf"
	"gopkg.in/sensorbee/sensorbee.v0/core"
	"gopkg.in/sensorbee/sensorbee.v0/data"
	"io"
	"reflect"
)

// regressionMsgpack has information of the saved file.
type regressionMsgpack struct {
	_struct   struct{} `codec:",toarray"`
	Algorithm string
}

type PassiveAggressiveState struct {
	pa                 *PassiveAggressive
	valueField         string
	featureVectorField string
}

var _ core.SavableSharedState = &PassiveAggressiveState{}

type paStateMsgpack struct {
	_struct            struct{} `codec:",toarray"`
	ValueField         string
	FeatureVectorField string
}

// PassiveAggressiveStateCreator is used by BQL to create PassiveAggressiveState as a UDS.
type PassiveAggressiveStateCreator struct {
}

var _ udf.UDSLoader = &PassiveAggressiveStateCreator{}

func (c *PassiveAggressiveStateCreator) CreateState(ctx *core.Context, params data.Map) (core.SharedState, error) {
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

var (
	regressionMsgpackHandle = &codec.MsgpackHandle{
		RawToString: true,
	}
)

func init() {
	regressionMsgpackHandle.MapType = reflect.TypeOf(map[string]interface{}{})
}

// LoadState loads a new state for PassiveAggressive model.
func (c *PassiveAggressiveStateCreator) LoadState(ctx *core.Context, r io.Reader, params data.Map) (core.SharedState, error) {
	formatVersion := make([]byte, 1)
	if _, err := r.Read(formatVersion); err != nil {
		return nil, err
	}

	switch formatVersion[0] {
	case 1:
		return loadPassiveAggressiveStateFormatV1(ctx, r)
	default:
		return nil, fmt.Errorf("unsupported format version of PassiveAggressiveState container: %v", formatVersion[0])
	}
}

func loadPassiveAggressiveStateFormatV1(ctx *core.Context, r io.Reader) (core.SharedState, error) {
	var header regressionMsgpack
	dec := codec.NewDecoder(r, regressionMsgpackHandle)
	if err := dec.Decode(&header); err != nil {
		return nil, err
	}
	if header.Algorithm != "passive_aggressive" {
		return nil, fmt.Errorf("unsupported regression algorithm: %v", header.Algorithm)
	}

	s := &PassiveAggressiveState{}

	var d paStateMsgpack
	if err := dec.Decode(&d); err != nil {
		return nil, err
	}
	s.valueField = d.ValueField
	s.featureVectorField = d.FeatureVectorField

	pa, err := LoadPassiveAggressive(r)
	if err != nil {
		return nil, err
	}
	s.pa = pa
	return s, nil
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

const (
	regressionFormatVersion = 1
)

// Save is provided as a part of core.SavableSharedState.
func (pa *PassiveAggressiveState) Save(ctx *core.Context, w io.Writer, params data.Map) error {
	if _, err := w.Write([]byte{regressionFormatVersion}); err != nil {
		return err
	}

	enc := codec.NewEncoder(w, regressionMsgpackHandle)
	if err := enc.Encode(&regressionMsgpack{
		Algorithm: "passive_aggressive",
	}); err != nil {
		return err
	}

	if err := enc.Encode(&paStateMsgpack{
		ValueField:         pa.valueField,
		FeatureVectorField: pa.featureVectorField,
	}); err != nil {
		return err
	}
	return pa.pa.Save(w)
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

package plugin

import (
	"fmt"
	"pfi/sensorbee/jubatus/anomaly"
	"pfi/sensorbee/jubatus/internal/pluginutil"
	"pfi/sensorbee/sensorbee/bql/udf"
	"pfi/sensorbee/sensorbee/core"
	"pfi/sensorbee/sensorbee/data"
	"strings"
)

func init() {
	if err := udf.RegisterGlobalUDSCreator("jubaanomaly_light_lof", udf.UDSCreatorFunc(newLightLOFState)); err != nil {
		panic(err)
	}

	if err := udf.RegisterGlobalUDF("jubaanomaly_add_and_get_score", udf.MustConvertGeneric(addAndGetScore)); err != nil {
		panic(err)
	}
}

type lightLOFState struct {
	lightLOF           *anomaly.LightLOF
	featureVectorField string
}

func newLightLOFState(ctx *core.Context, params data.Map) (core.SharedState, error) {
	fv, err := pluginutil.ExtractParamAsStringWithDefault(params, "feature_vector_field", "feature_vector")
	if err != nil {
		return nil, err
	}

	nnAlgoName, err := pluginutil.ExtractParamAsString(params, "nearest_neighbor_algorithm")
	if err != nil {
		return nil, err
	}

	var nnAlgo anomaly.NNAlgorithm
	switch strings.ToLower(nnAlgoName) {
	case "lsh":
		nnAlgo = anomaly.LSH
	case "minhash":
		nnAlgo = anomaly.Minhash
	case "euclid_lsh":
		nnAlgo = anomaly.EuclidLSH
	default:
		return nil, fmt.Errorf("invalid nearest_neighbor_algorithm: %s", nnAlgoName)
	}

	hashNum, err := pluginutil.ExtractParamAsInt(params, "hash_num")
	if err != nil {
		return nil, err
	}
	nnNum, err := pluginutil.ExtractParamAsInt(params, "nearest_neighbor_num")
	if err != nil {
		return nil, err
	}
	rnnNum, err := pluginutil.ExtractParamAsInt(params, "reverse_nearest_neighbor_num")
	if err != nil {
		return nil, err
	}

	// TODO: check hashNum, nnNum, rnnNum <= INT_MAX
	llof, err := anomaly.NewLightLOF(nnAlgo, int(hashNum), int(nnNum), int(rnnNum))
	if err != nil {
		return nil, err
	}
	return &lightLOFState{
		lightLOF:           llof,
		featureVectorField: fv,
	}, nil
}

func (*lightLOFState) Terminate(ctx *core.Context) error {
	return nil
}

func (l *lightLOFState) Write(ctx *core.Context, t *core.Tuple) error {
	vfv, ok := t.Data[l.featureVectorField]
	if !ok {
		return fmt.Errorf("%s field is missing", l.featureVectorField)
	}
	fv, err := data.AsMap(vfv)
	if err != nil {
		return fmt.Errorf("%s value is not a map: %v", l.featureVectorField, err)
	}

	_, err = l.lightLOF.AddWithoutCalcScore(anomaly.FeatureVector(fv))
	return err
}

func addAndGetScore(ctx *core.Context, stateName string, featureVector data.Map) (float32, error) {
	l, err := lookupLightLOFState(ctx, stateName)
	if err != nil {
		return 0, err
	}

	_, score, err := l.lightLOF.Add(anomaly.FeatureVector(featureVector))
	if err != nil {
		return 0, err
	}

	return score, nil
}

func lookupLightLOFState(ctx *core.Context, stateName string) (*lightLOFState, error) {
	st, err := ctx.SharedStates.Get(stateName)
	if err != nil {
		return nil, err
	}

	if l, ok := st.(*lightLOFState); ok {
		return l, nil
	}
	return nil, fmt.Errorf("state '%v' cannot be converted to lightLOFState", stateName)
}

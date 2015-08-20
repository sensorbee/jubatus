package plugin

import (
	"pfi/sensorbee/jubatus/anomaly"
	"pfi/sensorbee/sensorbee/bql/udf"
)

func init() {
	udf.MustRegisterGlobalUDSCreator("jubaanomaly_light_lof", udf.UDSCreatorFunc(anomaly.NewLightLOFState))

	udf.MustRegisterGlobalUDF("jubaanomaly_add_and_get_score", udf.MustConvertGeneric(anomaly.AddAndGetScore))
}

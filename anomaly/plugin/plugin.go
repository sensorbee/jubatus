package plugin

import (
	"github.com/sensorbee/jubatus/anomaly"
	"gopkg.in/sensorbee/sensorbee.v0/bql/udf"
)

func init() {
	udf.MustRegisterGlobalUDSCreator("jubaanomaly_light_lof", &anomaly.LightLOFStateCreator{})

	udf.MustRegisterGlobalUDF("jubaanomaly_add_and_get_score", udf.MustConvertGeneric(anomaly.AddAndGetScore))

	udf.MustRegisterGlobalUDF("jubaanomaly_calc_score", udf.MustConvertGeneric(anomaly.CalcScore))
}

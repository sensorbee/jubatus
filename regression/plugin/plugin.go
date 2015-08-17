package plugin

import (
	"pfi/sensorbee/jubatus/regression"
	"pfi/sensorbee/sensorbee/bql/udf"
)

func init() {
	udf.MustRegisterGlobalUDSCreator("jubaregression_pa", &regression.PassiveAggressiveStateCreator{})

	udf.MustRegisterGlobalUDF("jubaregression_estimate", udf.MustConvertGeneric(regression.PassiveAggressiveEstimate))
}

package plugin

import (
	"pfi/sensorbee/jubatus/regression"
	"pfi/sensorbee/sensorbee/bql/udf"
)

func init() {
	udf.MustRegisterGlobalUDSCreator("jubaregression_pa", &regression.PassiveAggressiveStateCreator{})
	if err := udf.RegisterGlobalUDF("jubaregression_estimate", udf.MustConvertGeneric(regression.PassiveAggressiveEstimate)); err != nil {
		panic(err)
	}
}

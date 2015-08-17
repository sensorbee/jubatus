package plugin

import (
	"pfi/sensorbee/jubatus/regression"
	"pfi/sensorbee/sensorbee/bql/udf"
)

func init() {
	if err := udf.RegisterGlobalUDSCreator("jubaregression_pa", udf.UDSCreatorFunc(regression.NewPassiveAggressiveState)); err != nil {
		panic(err)
	}
	if err := udf.RegisterGlobalUDF("jubaregression_estimate", udf.MustConvertGeneric(regression.PassiveAggressiveEstimate)); err != nil {
		panic(err)
	}
}

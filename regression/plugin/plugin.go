package plugin

import (
	"github.com/sensorbee/jubatus/regression"
	"gopkg.in/sensorbee/sensorbee.v0/bql/udf"
)

func init() {
	udf.MustRegisterGlobalUDSCreator("jubaregression_pa", &regression.PassiveAggressiveStateCreator{})

	udf.MustRegisterGlobalUDF("jubaregression_estimate", udf.MustConvertGeneric(regression.PassiveAggressiveEstimate))
}

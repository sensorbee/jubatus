package plugin

import (
	"github.com/sensorbee/jubatus/classifier"
	"github.com/sensorbee/jubatus/internal/math"
	"gopkg.in/sensorbee/sensorbee.v0/bql/udf"
)

func init() {
	udf.MustRegisterGlobalUDSCreator("jubaclassifier_arow", &classifier.AROWStateCreator{})

	// The name jubaclassify is not only for AROW, but for all classifier algorithms.
	// We have implemented only AROW, so we use the name for arowClassify for now.
	// When we have to implement another classification algorithm, generalize jubaclassify
	// to other algorithms. For example, define classifier.Classifier and adjust all algorithms to it.
	udf.MustRegisterGlobalUDF("jubaclassify", udf.MustConvertGeneric(classifier.AROWClassify))

	// TODO: consider to rename
	udf.MustRegisterGlobalUDF("juba_classified_label", udf.MustConvertGeneric(classifier.ClassifiedLabel))

	udf.MustRegisterGlobalUDF("juba_classified_score", udf.MustConvertGeneric(classifier.ClassifiedScore))
	udf.MustRegisterGlobalUDF("juba_softmax", udf.MustConvertGeneric(math.Softmax))
}

package plugin

import (
	"pfi/sensorbee/jubatus/classifier"
	"pfi/sensorbee/jubatus/internal/math"
	"pfi/sensorbee/sensorbee/bql/udf"
)

func init() {
	if err := udf.RegisterGlobalUDSCreator("jubaclassifier_arow", udf.UDSCreatorFunc(classifier.NewAROWState)); err != nil {
		panic(err)
	}

	// The name jubaclassify is not only for AROW, but for all classifier algorithms.
	// We have implemented only AROW, so we use the name for arowClassify for now.
	// When we have to implement another classification algorithm, generalize jubaclassify
	// to other algorithms. For example, define classifier.Classifier and adjust all algorithms to it.
	if err := udf.RegisterGlobalUDF("jubaclassify", udf.MustConvertGeneric(classifier.AROWClassify)); err != nil {
		panic(err)
	}

	// TODO: consider to rename
	if err := udf.RegisterGlobalUDF("juba_classified_label", udf.MustConvertGeneric(classifier.ClassifiedLabel)); err != nil {
		panic(err)
	}

	if err := udf.RegisterGlobalUDF("juba_classified_score", udf.MustConvertGeneric(classifier.ClassifiedScore)); err != nil {
		panic(err)
	}

	if err := udf.RegisterGlobalUDF("juba_softmax", udf.MustConvertGeneric(math.Softmax)); err != nil {
		panic(err)
	}
}

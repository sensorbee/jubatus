package regression

import (
	"bytes"
	. "github.com/smartystreets/goconvey/convey"
	"gopkg.in/sensorbee/sensorbee.v0/core"
	"gopkg.in/sensorbee/sensorbee.v0/data"
	"testing"
)

func TestPassiveAggressiveStateLoad(t *testing.T) {
	ctx := core.NewContext(nil)
	c := PassiveAggressiveStateCreator{}
	pas, err := c.CreateState(ctx, data.Map{
		"regularization_weight": data.Float(3.402823e+38),
		"sensitivity":           data.Float(0.1),
	})
	if err != nil {
		t.Fatal(err)
	}
	pa := pas.(*PassiveAggressiveState)

	for i := 0; i < 100; i++ {
		if err := pa.Write(ctx, &core.Tuple{
			Data: data.Map{
				"value": data.Float(i),
				"feature_vector": data.Map{
					"n": data.Int(i),
				},
			},
		}); err != nil {
			t.Fatal(err)
		}
	}

	Convey("Given a trained PassiveAggressiveState", t, func() {
		Convey("when saving it", func() {
			buf := bytes.NewBuffer(nil)
			err := pa.Save(ctx, buf, data.Map{})

			Convey("it should succeed.", func() {
				So(err, ShouldBeNil)

				Convey("and the loaded state should be same.", func() {
					pa2, err := c.LoadState(ctx, buf, data.Map{})
					So(err, ShouldBeNil)

					So(pa2, ShouldResemble, pa)

					fv := FeatureVector{
						"n": data.Int(123),
					}
					v, err := pa.pa.Estimate(fv)
					So(err, ShouldBeNil)
					v2, err := pa2.(*PassiveAggressiveState).pa.Estimate(fv)
					So(err, ShouldBeNil)
					So(v2, ShouldResemble, v)
				})
			})
		})
	})
}

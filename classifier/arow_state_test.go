package classifier

import (
	"bytes"
	. "github.com/smartystreets/goconvey/convey"
	"gopkg.in/sensorbee/sensorbee.v0/core"
	"gopkg.in/sensorbee/sensorbee.v0/data"
	"testing"
)

func TestAROWStateSaveLoad(t *testing.T) {
	ctx := core.NewContext(nil)
	c := AROWStateCreator{}
	as, err := c.CreateState(ctx, data.Map{
		"regularization_weight": data.Float(0.001),
	})
	if err != nil {
		t.Fatal(err)
	}
	a := as.(*AROWState)

	labels := []data.String{"a", "b", "c", "d"}
	for i := 0; i < 100; i++ {
		if err := a.Write(ctx, &core.Tuple{
			Data: data.Map{
				"label": labels[i%len(labels)],
				"feature_vector": data.Map{
					"n": data.Int(i),
				},
			},
		}); err != nil {
			t.Fatal(err)
		}
	}

	Convey("Given a trained AROWState", t, func() {
		Convey("when saving it", func() {
			buf := bytes.NewBuffer(nil)
			err := a.Save(ctx, buf, data.Map{})

			Convey("it should succeed.", func() {
				So(err, ShouldBeNil)

				Convey("and the loaded state should be same.", func() {
					a2, err := c.LoadState(ctx, buf, data.Map{})
					So(err, ShouldBeNil)

					// Because AROW contains sync.RWMutex, this assertion may
					// fail if its implementation changes.
					So(a2, ShouldResemble, a)

					fv := FeatureVector(data.Map{"n": data.Int(10)})
					s, err := a.arow.Classify(fv)
					So(err, ShouldBeNil)
					s2, err := a2.(*AROWState).arow.Classify(fv)
					So(err, ShouldBeNil)
					So(s2, ShouldResemble, s)
				})
			})
		})
	})
}

package anomaly

import (
	"bytes"
	. "github.com/smartystreets/goconvey/convey"
	"pfi/sensorbee/sensorbee/core"
	"pfi/sensorbee/sensorbee/data"
	"testing"
)

func TestLightLOFStaateSaveLoad(t *testing.T) {
	ctx := core.NewContext(nil)
	c := LightLOFStateCreator{}
	ls, err := c.CreateState(ctx, data.Map{
		"nearest_neighbor_algorithm":   data.String("minhash"),
		"hash_num":                     data.Int(64),
		"nearest_neighbor_num":         data.Int(10),
		"reverse_nearest_neighbor_num": data.Int(30),
	})
	if err != nil {
		t.Fatal(err)
	}
	l := ls.(*lightLOFState)

	for i := 0; i < 100; i++ {
		if err := l.Write(ctx, &core.Tuple{
			Data: data.Map{
				"feature_vector": data.Map{
					"n": data.Int(i),
				},
			},
		}); err != nil {
			t.Fatal(err)
		}
	}

	Convey("Given a trained LightLOFState", t, func() {
		Convey("when saving it", func() {
			buf := bytes.NewBuffer(nil)
			err := l.Save(ctx, buf, data.Map{})

			Convey("it should succeed.", func() {
				So(err, ShouldBeNil)

				Convey("and the loaded state should be same.", func() {
					l2, err := c.LoadState(ctx, buf, data.Map{})
					So(err, ShouldBeNil)

					m := l.lightLOF
					m2 := l2.(*lightLOFState).lightLOF
					So(m2.nn, ShouldResemble, m.nn)
					So(m2.nnNum, ShouldEqual, m.nnNum)
					So(m2.rnnNum, ShouldEqual, m.rnnNum)
					So(m2.kdists, ShouldResemble, m.kdists)
					So(m2.lrds, ShouldResemble, m.lrds)
					So(m2.maxSize, ShouldEqual, m.maxSize)
					So(m2.rg, ShouldNotBeNil)

					fv := FeatureVector(data.Map{"n": data.Int(10)})
					s, err := l.lightLOF.CalcScore(fv)
					So(err, ShouldBeNil)
					s2, err := l2.(*lightLOFState).lightLOF.CalcScore(fv)
					So(err, ShouldBeNil)
					So(s2, ShouldResemble, s)
				})
			})
		})
	})
}

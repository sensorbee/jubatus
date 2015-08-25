package bitvector

import (
	"fmt"
	. "github.com/smartystreets/goconvey/convey"
	"testing"
)

func TestArray(t *testing.T) {
	for i := 0; i < 100; i++ {
		bitNum := i + 1
		v := NewVector(bitNum)

		Convey(fmt.Sprint("Given a %v-bit vector", bitNum), t, func() {
			Convey("it should not be nil.", func() {
				So(v, ShouldNotBeNil)
			})

			Convey("when setting the least significant bit", func() {
				err := v.Set(0)
				Convey("it should succeed.", func() {
					So(err, ShouldBeNil)
				})

				Convey("the least significant bit should be set.", func() {
					So(v.GetAsUint64(0), ShouldEqual, 1)
				})
			})
		})
	}
}

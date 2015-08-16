package intern

import (
	"bytes"
	. "github.com/smartystreets/goconvey/convey"
	"testing"
)

func TestIntern(t *testing.T) {
	Convey("Given an Intern", t, func() {
		i := New()

		Convey("when getting an ID of a key", func() {
			id := i.Get("a")

			Convey("it shouldn't be zero", func() {
				So(id, ShouldNotEqual, 0)
			})

			Convey("getting the ID of the key again should return the same value", func() {
				So(i.Get("a"), ShouldEqual, id)
				So(i.GetOrZero("a"), ShouldEqual, id)
			})

			Convey("and getting an ID of another key", func() {
				id2 := i.Get("b")

				Convey("two IDs shouldn't be same", func() {
					So(id, ShouldNotEqual, id2)
				})
			})
		})

		Convey("when getting an ID of a nonexistent key with GetOrZero", func() {
			id := i.GetOrZero("a")

			Convey("it should be zero", func() {
				So(id, ShouldEqual, 0)
			})
		})
	})
}

func TestInternSaveLoad(t *testing.T) {
	i := New()
	i.Get("a")
	i.Get("b")
	i.Get("c")
	i.Get("d")
	i.Get("e")

	Convey("Given an Intern with keys", t, func() {
		Convey("when saving it", func() {
			buf := bytes.NewBuffer(nil)
			err := i.Save(buf)

			Convey("it should succeed", func() {
				So(err, ShouldBeNil)

				Convey("and the data should be able to be loaded", func() {
					i2, err := Load(buf)
					So(err, ShouldBeNil)
					So(i2, ShouldResemble, i)
				})
			})
		})
	})
}

package nested

import (
	. "github.com/smartystreets/goconvey/convey"
	"gopkg.in/sensorbee/sensorbee.v0/data"
	"testing"
)

type kv struct {
	key   string
	value float32
}

func TestFlattenEmptyMap(t *testing.T) {
	m := data.Map{}

	Convey("Given an empty data.Map", t, func() {
		Convey("when flatten it", func() {
			cnt := 0
			err := Flatten(m, func(string, float32) {
				cnt++
			})

			Convey("it should succeed.", func() {
				So(err, ShouldBeNil)

				Convey("and the appender should not be called.", func() {
					So(cnt, ShouldBeZeroValue)
				})
			})
		})
	})
}

func TestFlattenFlatMap(t *testing.T) {
	m := data.Map{
		"abc": data.Float(123),
		"def": data.Float(456),
		"ghi": data.Float(789),
	}

	Convey("Given a flat data.Map", t, func() {
		Convey("when flatten it", func() {
			a := []*kv{}
			err := Flatten(m, func(k string, x float32) {
				a = append(a, &kv{
					key:   k,
					value: x,
				})
			})

			Convey("it should succeed.", func() {
				So(err, ShouldBeNil)

				Convey("and the flatten slice should be converted correctly.", func() {
					So(len(a), ShouldEqual, len(m))
					for _, e := range a {
						mValue, ok := m[e.key]
						So(ok, ShouldBeTrue)
						So(e.value, ShouldEqual, mValue)
					}
				})
			})
		})
	})
}

func TestFlattenNestedEmptyMap(t *testing.T) {
	m := data.Map{
		"a": data.Map{},
		"b": data.Map{},
		"c": data.Map{
			"d": data.Map{
				"e": data.Map{
					"f": data.Map{
						"g": data.Map{},
					},
				},
			},
		},
	}

	Convey("Given a nested empty data.Map", t, func() {
		cnt := 0
		err := Flatten(m, func(string, float32) {
			cnt++
		})

		Convey("it should succeed.", func() {
			So(err, ShouldBeNil)

			Convey("and the appender should not be called.", func() {
				So(cnt, ShouldBeZeroValue)
			})
		})
	})
}

func TestFlattenNestedMap(t *testing.T) {
	m := data.Map{
		"a": data.Float(123),
		"b": data.Map{
			"c": data.Float(456),
			"d": data.Map{},
			"e": data.Float(789),
		},
		"f": data.Map{},
		"g": data.Float(1234),
		"h": data.Map{
			"i": data.Map{
				"j": data.Map{
					"k": data.Map{
						"l": data.Map{
							"m": data.Map{
								"n": data.Map{
									"o": data.Float(5678),
								},
							},
						},
					},
				},
			},
		},
	}
	flattenM := data.Map{
		"a":      data.Float(123),
		"b\x00c": data.Float(456),
		"b\x00e": data.Float(789),
		"g":      data.Float(1234),
		"h\x00i\x00j\x00k\x00l\x00m\x00n\x00o": data.Float(5678),
	}

	Convey("Given a nested data.Map", t, func() {
		a := []*kv{}
		err := Flatten(m, func(k string, x float32) {
			a = append(a, &kv{
				key:   k,
				value: x,
			})
		})

		Convey("it should succeed.", func() {
			So(err, ShouldBeNil)

			Convey("and the flatten slice should be converted correctly.", func() {
				So(len(a), ShouldEqual, len(flattenM))
				for _, e := range a {
					mValue, ok := flattenM[e.key]
					So(ok, ShouldBeTrue)
					So(e.value, ShouldEqual, mValue)
				}
			})
		})
	})
}

func TestFlattenEmptyArray(t *testing.T) {
	m := data.Map{
		"a": data.Array{},
	}

	Convey("Given a data.Map having an empty data.Array", t, func() {
		Convey("when flatten it", func() {
			cnt := 0
			err := Flatten(m, func(string, float32) {
				cnt++
			})

			Convey("it should succeed.", func() {
				So(err, ShouldBeNil)

				Convey("and the appender should not be called.", func() {
					So(cnt, ShouldBeZeroValue)
				})
			})
		})
	})
}

func TestFlattenNonNestedArray(t *testing.T) {
	m := data.Map{
		"a": data.Array{data.Float(123), data.Float(456), data.Float(789)},
	}

	Convey("Given a flat data.Map having a data.Array", t, func() {
		Convey("when flatten it", func() {
			a := []*kv{}
			err := Flatten(m, func(k string, x float32) {
				a = append(a, &kv{
					key:   k,
					value: x,
				})
			})

			Convey("it should succeed.", func() {
				So(err, ShouldBeNil)

				Convey("and the flatten slice should be converted correctly.", func() {
					So(len(a), ShouldEqual, 3)
				})
			})
		})
	})
}

func TestFlattenNestedArray(t *testing.T) {
	m := data.Map{
		"a": data.Array{data.Array{}, data.Array{data.Float(123), data.Float(456), data.Float(789)}},
	}

	Convey("Given a flat data.Map having a data.Array", t, func() {
		Convey("when flatten it", func() {
			a := []*kv{}
			err := Flatten(m, func(k string, x float32) {
				a = append(a, &kv{
					key:   k,
					value: x,
				})
			})

			Convey("it should succeed.", func() {
				So(err, ShouldBeNil)

				Convey("and the flatten slice should be converted correctly.", func() {
					So(len(a), ShouldEqual, 3)
				})
			})
		})
	})
}

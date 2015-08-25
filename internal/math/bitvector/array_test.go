package bitvector

import (
	"bytes"
	"fmt"
	. "github.com/smartystreets/goconvey/convey"
	"testing"
)

func TestArrayOfOneBit(t *testing.T) {
	a := NewArray(1)

	Convey("Given an arry of one bit", t, func() {
		Convey("it's initial size should be zero.", func() {
			So(a.Len(), ShouldBeZeroValue)
		})

		Convey("when resize it to contain ten elements", func() {
			a.Resize(10)
			Convey("it's size shoulde be ten", func() {
				So(a.Len(), ShouldEqual, 10)
			})

			Convey("and all bits should be zero.", func() {
				for j := 0; j < a.Len(); j++ {
					v, err := a.Get(j)
					So(err, ShouldBeNil)
					So(v, ShouldNotBeNil)
					So(v.GetAsUint64(0), ShouldBeZeroValue)
				}
			})
		})

		Convey("when setting one to the third element", func() {
			one := NewVector(1)
			one.Set(0)

			a.Set(2, one)
			Convey("the third element should be one", func() {
				third, err := a.Get(2)
				So(err, ShouldBeNil)
				So(third.GetAsUint64(0), ShouldEqual, 1)
			})

			Convey("and other elements should be zero.", func() {
				for i := 0; i < a.Len(); i++ {
					if i != 2 {
						v, err := a.Get(i)
						So(err, ShouldBeNil)
						So(v.GetAsUint64(0), ShouldBeZeroValue)
					}
				}
			})
		})
	})
}

func TestArrayOfSmallBits(t *testing.T) {
	// 2, 3, 4, ..., wordBits
	for i := 2; i <= wordBits; i++ {
		bitNum := i
		a := NewArray(bitNum)

		Convey(fmt.Sprintf("Given an array of %v bits", bitNum), t, func() {
			Convey("it's initial size should be zero.", func() {
				So(a.Len(), ShouldBeZeroValue)
			})

			Convey("when resizing it to contain ten elements", func() {
				a.Resize(10)
				Convey("it's size shoulde be ten", func() {
					So(a.Len(), ShouldEqual, 10)
				})

				Convey("and all bits should be zero.", func() {
					for j := 0; j < a.Len(); j++ {
						v, err := a.Get(j)
						So(err, ShouldBeNil)
						So(v.GetAsUint64(0), ShouldBeZeroValue)
					}
				})
			})

			Convey("when setting one to the third element", func() {
				one := NewVector(bitNum)
				one.Set(0)
				a.Set(2, one)
				Convey("the third element should be one", func() {
					third, err := a.Get(2)
					So(err, ShouldBeNil)
					So(third.GetAsUint64(0), ShouldEqual, 1)
				})

				Convey("and other elements should be zero.", func() {
					for i := 0; i < a.Len(); i++ {
						if i != 2 {
							v, err := a.Get(i)
							So(err, ShouldBeNil)
							So(v.GetAsUint64(0), ShouldBeZeroValue)
						}
					}
				})
			})

			Convey("when setting least and most significant bits to the third element", func() {
				bits := NewVector(bitNum)
				bits.Set(0)
				bits.Set(bitNum - 1)
				a.Set(2, bits)
				Convey("the third element should be the bits", func() {
					third, err := a.Get(2)
					So(err, ShouldBeNil)
					So(third.GetAsUint64(0), ShouldEqual, bits.GetAsUint64(0))
				})

				Convey("and other elements should be zero.", func() {
					for i := 0; i < a.Len(); i++ {
						if i != 2 {
							v, err := a.Get(i)
							So(err, ShouldBeNil)
							So(v.GetAsUint64(0), ShouldBeZeroValue)
						}
					}
				})
			})

			Convey("when setting all bits to the third element", func() {
				bits := NewVector(bitNum)
				for j := 0; j < bitNum; j++ {
					bits.Set(j)
				}
				a.Set(2, bits)
				Convey("the third element should be the bits", func() {
					third, err := a.Get(2)
					So(err, ShouldBeNil)
					So(third.GetAsUint64(0), ShouldEqual, bits.GetAsUint64(0))
				})

				Convey("and other elements should be zero.", func() {
					for i := 0; i < a.Len(); i++ {
						if i != 2 {
							v, err := a.Get(i)
							So(err, ShouldBeNil)
							So(v.GetAsUint64(0), ShouldBeZeroValue)
						}
					}
				})
			})
		})
	}
}

func TestArrayOfLargePrimeBits(t *testing.T) {
	primes := []int{
		67, 71, 73, 79, 83, 89, 97, 101,
		103, 107, 109, 113, 127, 131, 137, 139,
		149, 151, 157, 163, 167, 173, 179, 181,
		191, 193, 197, 199, 211, 223, 227, 229,
		233, 239, 241, 251, 257, 263, 269, 271,
		277, 281, 283, 293, 307, 311, 313, 317,
		331, 337, 347, 349, 353, 359, 367, 373,
		379, 383, 389, 397, 401, 409, 419, 421,
		431, 433, 439, 443, 449, 457, 461, 463,
		467, 479, 487, 491, 499, 503, 509, 521,
	}

	for _, bitNum := range primes {
		a := NewArray(bitNum)
		nw := nWords(bitNum, 1)

		Convey(fmt.Sprintf("Given an array of %v bits", bitNum), t, func() {
			Convey("it's initial size should be zero.", func() {
				So(a.Len(), ShouldBeZeroValue)
			})

			Convey("when resizing it to contain ten elements", func() {
				a.Resize(10)
				Convey("it's size shoulde be ten", func() {
					So(a.Len(), ShouldEqual, 10)
				})

				Convey("and all bits should be zero.", func() {
					for i := 0; i < a.Len(); i++ {
						v, err := a.Get(i)
						So(err, ShouldBeNil)
						for j := 0; j < nw; j++ {
							So(v.GetAsUint64(j), ShouldBeZeroValue)
						}
					}
				})
			})

			Convey("when setting one to the third element", func() {
				one := NewVector(bitNum)
				one.Set(0)
				a.Set(2, one)
				Convey("the third element should be one", func() {
					third, err := a.Get(2)
					So(err, ShouldBeNil)
					So(third.GetAsUint64(0), ShouldEqual, 1)
					for i := 1; i < nw; i++ {
						So(third.GetAsUint64(i), ShouldBeZeroValue)
					}
				})

				Convey("and other elements should be zero.", func() {
					for i := 0; i < a.Len(); i++ {
						if i != 2 {
							v, err := a.Get(i)
							So(err, ShouldBeNil)
							for j := 0; j < nw; j++ {
								So(v.GetAsUint64(j), ShouldBeZeroValue)
							}
						}
					}
				})
			})

			Convey("when setting least and most significant bits to the third element", func() {
				bits := NewVector(bitNum)
				bits.Set(0)
				bits.Set(bitNum - 1)
				a.Set(2, bits)
				Convey("the third element should be the bits", func() {
					third, err := a.Get(2)
					So(err, ShouldBeNil)
					for i := 0; i < nw; i++ {
						So(third.GetAsUint64(i), ShouldEqual, bits.GetAsUint64(i))
					}
				})

				Convey("and other elements should be zero.", func() {
					for i := 0; i < a.Len(); i++ {
						if i != 2 {
							v, err := a.Get(i)
							So(err, ShouldBeNil)
							for j := 0; j < nw; j++ {
								So(v.GetAsUint64(j), ShouldBeZeroValue)
							}
						}
					}
				})
			})

			Convey("when setting all bits to the third element", func() {
				bits := NewVector(bitNum)
				for j := 0; j < bitNum; j++ {
					bits.Set(j)
				}
				a.Set(2, bits)
				Convey("the third element should be the bits", func() {
					third, err := a.Get(2)
					So(err, ShouldBeNil)
					for i := 0; i < nw; i++ {
						So(third.GetAsUint64(i), ShouldEqual, bits.GetAsUint64(i))
					}
				})

				Convey("and other elements should be zero.", func() {
					for i := 0; i < a.Len(); i++ {
						if i != 2 {
							v, err := a.Get(i)
							So(err, ShouldBeNil)
							for j := 0; j < nw; j++ {
								So(v.GetAsUint64(j), ShouldBeZeroValue)
							}
						}
					}
				})
			})
		})
	}
}

func TestArraySaveLoad(t *testing.T) {
	a := NewArray(10)
	a.Resize(10)
	for i := 0; i < a.Len(); i++ {
		v := NewVector(10)
		v.Set(i)
		a.Set(i, v)
	}

	Convey("Given an Array", t, func() {
		Convey("when saving it", func() {
			buf := bytes.NewBuffer(nil)
			err := a.Save(buf)

			Convey("it should succeed", func() {
				So(err, ShouldBeNil)

				Convey("and the data should be loaded", func() {
					a2, err := LoadArray(buf)
					So(err, ShouldBeNil)
					So(a2, ShouldResemble, a)
				})
			})
		})
	})
}

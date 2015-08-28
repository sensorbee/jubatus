package bit

import (
	. "github.com/smartystreets/goconvey/convey"
	"math/rand"
	"testing"
)

func TestCount(t *testing.T) {
	Convey("Given zero", t, func() {
		Convey("it's bitcount should be zero.", func() {
			So(bitcount(0), ShouldBeZeroValue)
		})
	})

	Convey("Given words with one 1", t, func() {
		Convey("it's bitcount should be one.", func() {
			for i := uint(0); i < wordBits; i++ {
				So(bitcount(1<<i), ShouldEqual, 1)
			}
		})
	})

	Convey("Given words with two 1's", t, func() {
		Convey("it's bitcount should be two.", func() {
			for i := uint(0); i < wordBits-1; i++ {
				var w word = 1 | (1 << (i + 1))
				for j := uint(0); j < wordBits-i-2; j++ {
					w <<= 1
					So(bitcount(w), ShouldEqual, 2)
				}
			}
		})
	})

	Convey("Given words with thirty seven 1's", t, func() {
		Convey("it's bitcount should be thirty seven.", func() {
			r := rand.New(rand.NewSource(0))
			for i := 0; i < 10000; i++ {
				w := randomBits(r, 37, wordBits)
				So(bitcount(w), ShouldEqual, 37)
			}
		})
	})

	Convey("Given words with full bits", t, func() {
		Convey("it's bitcount should be it's bit number.", func() {
			w := ^word(0)
			So(bitcount(w), ShouldEqual, wordBits)
		})
	})
}

func TestCount32(t *testing.T) {
	Convey("Given zero", t, func() {
		Convey("it's bitcount should be zero.", func() {
			So(bitcount32(0), ShouldBeZeroValue)
		})
	})

	Convey("Given uint32s with one 1", t, func() {
		Convey("it's bitcount should be one.", func() {
			for i := uint(0); i < 32; i++ {
				So(bitcount32(1<<i), ShouldEqual, 1)
			}
		})
	})

	Convey("Given uint32s with two 1's", t, func() {
		Convey("it's bitcount should be two.", func() {
			for i := uint(0); i < 32-1; i++ {
				var x uint32 = 1 | (1 << (i + 1))
				for j := uint(0); j < 32-i-2; j++ {
					x <<= 1
					So(bitcount32(x), ShouldEqual, 2)
				}
			}
		})
	})

	Convey("Given uint32s with twenty three 1's", t, func() {
		Convey("it's bitcount should be twenty three.", func() {
			r := rand.New(rand.NewSource(0))
			for i := 0; i < 10000; i++ {
				x := uint32(randomBits(r, 23, 32))
				So(bitcount32(x), ShouldEqual, 23)
			}
		})
	})

	Convey("Given uint32s with full bits", t, func() {
		Convey("it's bitcount should be it's bit number.", func() {
			x := ^uint32(0)
			So(bitcount32(x), ShouldEqual, 32)
		})
	})
}

func TestCount16(t *testing.T) {
	Convey("Given zero", t, func() {
		Convey("it's bitcount should be zero.", func() {
			So(bitcount16(0), ShouldBeZeroValue)
		})
	})

	Convey("Given uint16s with one 1", t, func() {
		Convey("it's bitcount should be one.", func() {
			for i := uint(0); i < 16; i++ {
				So(bitcount16(1<<i), ShouldEqual, 1)
			}
		})
	})

	Convey("Given uint16s with two 1's", t, func() {
		Convey("it's bitcount should be two.", func() {
			for i := uint(0); i < 16-1; i++ {
				var x uint16 = 1 | (1 << (i + 1))
				for j := uint(0); j < 16-i-2; j++ {
					x <<= 1
					So(bitcount16(x), ShouldEqual, 2)
				}
			}
		})
	})

	Convey("Given uint16s with eleven 1's", t, func() {
		Convey("it's bitcount should be eleven.", func() {
			r := rand.New(rand.NewSource(0))
			for i := 0; i < 10000; i++ {
				x := uint16(randomBits(r, 11, 16))
				So(bitcount16(x), ShouldEqual, 11)
			}
		})
	})

	Convey("Given uint16s with full bits", t, func() {
		Convey("it's bitcount should be it's bit number.", func() {
			x := ^uint16(0)
			So(bitcount16(x), ShouldEqual, 16)
		})
	})
}

func randomBits(r *rand.Rand, nOneBits, nAllBits int) word {
	if nOneBits > nAllBits {
		panic("nOneBits > nAllBits")
	}
	if nAllBits > wordBits {
		panic("nAllBits > wordBits")
	}
	if nOneBits == nAllBits {
		return (1 << uint(nAllBits)) - 1
	}
	var w word
	for nOneBits > 0 {
		var b word = 1 << uint(r.Intn(nAllBits))
		if w&b == 0 {
			w |= b
			nOneBits--
		}
	}
	return w
}

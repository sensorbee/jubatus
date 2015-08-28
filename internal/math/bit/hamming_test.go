package bit

import (
	"fmt"
	. "github.com/smartystreets/goconvey/convey"
	"math/rand"
	"testing"
)

func TestHamming(t *testing.T) {
	Convey("Given bitarrays with all zero bits", t, func() {
		for i := 1; i <= 129; i++ {
			a := NewArray(i)
			a.Resize(10)

			Convey(fmt.Sprintf("hamming distance between a %v bit zero vector and a zero vector should be zero.", i), func() {
				v := NewVector(i)
				for j := 0; j < a.Len(); j++ {
					hd, err := a.HammingDistance(j, v)
					So(err, ShouldBeNil)
					So(hd, ShouldBeZeroValue)
				}
			})

			Convey(fmt.Sprintf("hamming distance between a %v bit zero vector with a vector with one 1 should be 1.", i), func() {
				for j := 0; j < i; j++ {
					v := NewVector(i)
					v.Set(j)
					for k := 0; k < a.Len(); k++ {
						hd, err := a.HammingDistance(k, v)
						So(err, ShouldBeNil)
						So(hd, ShouldEqual, 1)
					}
				}
			})

			Convey(fmt.Sprintf("hamming distance between a %v bit zero vector with a vector with full bits should be %v.", i, i), func() {
				v := NewVector(i)
				for j := 0; j < i; j++ {
					v.Set(j)
				}
				for j := 0; j < a.Len(); j++ {
					hd, err := a.HammingDistance(j, v)
					So(err, ShouldBeNil)
					So(hd, ShouldEqual, i)
				}
			})
		}
	})

	Convey("Given bitarrays with random bits", t, func() {
		r := rand.New(rand.NewSource(0))
		for i := 1; i <= 129; i++ {
			a := GenerateRandomArray(r, 10, i)

			Convey(fmt.Sprint("hamming distance between a %v bit vector and itself should be zero.", i), func() {
				for j := 0; j < a.Len(); j++ {
					v, _ := a.Get(j)
					hd, err := a.HammingDistance(j, v)
					So(err, ShouldBeNil)
					So(hd, ShouldEqual, 0)
				}
			})

			Convey(fmt.Sprint("hamming distance between %v bit vectors with one distance should be one.", i), func() {
				for j := 0; j < a.Len(); j++ {
					v, _ := a.Get(j)
					v = GenerateVectorWithHammingDistance(r, v, 1)
					hd, err := a.HammingDistance(j, v)
					So(err, ShouldBeNil)
					So(hd, ShouldEqual, 1)
				}
			})

			Convey(fmt.Sprint("hamming distance between %v bit vectors with %v distance should be %v.", i, i/2, i/2), func() {
				for j := 0; j < a.Len(); j++ {
					v, _ := a.Get(j)
					v = GenerateVectorWithHammingDistance(r, v, i/2)
					hd, err := a.HammingDistance(j, v)
					So(err, ShouldBeNil)
					So(hd, ShouldEqual, i/2)
				}
			})

			Convey(fmt.Sprint("hamming distance between %v bit vectors with %v distance should be %v.", i, i, i), func() {
				for j := 0; j < a.Len(); j++ {
					v, _ := a.Get(j)
					v = GenerateVectorWithHammingDistance(r, v, i)
					hd, err := a.HammingDistance(j, v)
					So(err, ShouldBeNil)
					So(hd, ShouldEqual, i)
				}
			})
		}
	})
}

func GenerateVectorWithHammingDistance(r *rand.Rand, v *Vector, hd int) *Vector {
	if v.bitNum < hd {
		panic("v.bitNum < hd")
	}
	if hd < 0 {
		panic("hd < 0")
	}

	bits := map[int]bool{}
	for hd > 0 {
		b := r.Intn(v.bitNum)
		if !bits[b] {
			bits[b] = true
			hd--
		}
	}
	ret := NewVector(v.bitNum)
	copy(ret.data, v.data)
	for b := range bits {
		ret.reverse(b)
	}
	return ret
}

func GenerateRandomArray(r *rand.Rand, len, bitNum int) Array {
	a := NewArray(bitNum)
	a.Resize(len)
	for i := 0; i < len; i++ {
		a.Set(i, GenerateRandomVector(r, bitNum))
	}
	return a
}

func GenerateRandomVector(r *rand.Rand, bitNum int) *Vector {
	v := NewVector(bitNum)
	for i := 0; i < bitNum; i++ {
		if r.Intn(2) == 1 {
			v.Set(i)
		}
	}
	return v
}

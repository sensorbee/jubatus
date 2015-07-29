package math

import (
	"fmt"
	"pfi/sensorbee/sensorbee/data"
)

func Example() {
	v := data.Map{
		"labelA": data.Float(2.5),
		"labelB": data.Float(0.7),
		"labelC": data.Float(-1.2),
	}
	s, _ := Softmax(v)
	fmt.Printf("labelA: %0.5f\n", toFloat(s["labelA"]))
	fmt.Printf("labelB: %0.5f\n", toFloat(s["labelB"]))
	fmt.Printf("labelC: %0.5f\n", toFloat(s["labelC"]))

	// Output:
	// labelA: 0.84032
	// labelB: 0.13890
	// labelC: 0.02078
}

func toFloat(d data.Value) float64 {
	ret, _ := data.AsFloat(d)
	return ret
}

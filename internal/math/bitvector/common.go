package bitvector

// word is uint64 for serialization.
// This may be changed to uintptr.
type word uint64
type buf []word

func nWords(bitNum, len int) int {
	return (bitNum*len + wordBits - 1) / wordBits
}

func leastBits(n int) word {
	if n >= wordBits {
		return maxWordValue
	}

	return (1 << word(n)) - 1
}

const (
	maxWordValue = ^word(0)
	// shorthand
	m = maxWordValue
	// expect uintptr is (8|16|32|64)-bit integer type.
	// see math/big/arith.go in the standard library.
	wordBits = 1 << ((m>>8&1 + m>>16&1 + m>>32&1) + 3)
)

func maxInt(x, y int) int {
	if x < y {
		return y
	}
	return x
}

func set(w *word, offset int, x word, nbits int) {
	mask := leastBits(nbits)
	*w = (*w & ^(mask << uint(offset))) | ((x & mask) << uint(offset))
}

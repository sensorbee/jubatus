package bit

func bitcount(x word) int {
	x -= (x >> 1) & 0x5555555555555555
	x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
	x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F
	x += x >> 8
	x += x >> 16
	x += x >> 32
	return int(x & 0x7F)
}

func bitcount32(x uint32) int {
	n := (x >> 1) & 033333333333
	x -= n
	n = (n >> 1) & 033333333333
	x -= n
	x = (x + (x >> 3)) & 030707070707
	return int(x % 63)
}

func bitcount32s(x uint64) (int, int) {
	x -= (x >> 1) & 0x5555555555555555
	x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
	x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F
	x += x >> 8
	x += x >> 16
	return int(x & 0x3F), int((x >> 32) & 0x3F)
}

func bitcount16(x uint16) int {
	x -= (x >> 1) & 0x5555
	x = (x & 0x3333) + ((x >> 2) & 0x3333)
	x = (x + (x >> 4)) & 0x0F0F
	x += x >> 8
	return int(x & 0x1F)
}

func bitcount8(x uint8) int {
	x -= (x >> 1) & 0x55
	x = (x & 0x33) + ((x >> 2) & 0x33)
	x += x >> 4
	return int(x & 0xF)
}

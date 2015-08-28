package bit

func bitcount(x word) int {
	return int(bitcountTable[x&bitcountMask]) + int(bitcountTable[x>>16&bitcountMask]) +
		int(bitcountTable[x>>32&bitcountMask]) + int(bitcountTable[x>>48])
}

func bitcount32(x uint32) int {
	return int(bitcountTable[x&uint32(bitcountMask)]) + int(bitcountTable[x>>16])
}

func bitcount16(x uint16) int {
	return int(bitcountTable[x])
}

const bitcountMask = word(^uint16(0))

var bitcountTable = [bitcountMask + 1]uint8{}

func init() {
	for i := range bitcountTable {
		var cnt uint8
		n := i
		for n != 0 {
			cnt++
			n &= n - 1
		}
		bitcountTable[i] = cnt
	}
}

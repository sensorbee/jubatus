package classifier

// TODO: consider where this file should be placed

type intern struct {
	storage map[string]int
	gen     int
}

func newIntern() *intern {
	return &intern{
		storage: make(map[string]int),
		gen:     0,
	}
}

func (i *intern) mayGet(s string) int {
	return i.storage[s]
}

func (i *intern) get(s string) int {
	if i.mayGet(s) == 0 {
		i.gen++
		i.storage[s] = i.gen
		return i.gen
	}
	return i.storage[s]
}

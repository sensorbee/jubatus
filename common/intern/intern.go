package intern

type Intern struct {
	storage map[string]int
	gen     int
}

func New() *Intern {
	return &Intern{
		storage: make(map[string]int),
		gen:     0,
	}
}

func (i *Intern) MayGet(s string) int {
	return i.storage[s]
}

func (i *Intern) Get(s string) int {
	if i.MayGet(s) == 0 {
		i.gen++
		i.storage[s] = i.gen
		return i.gen
	}
	return i.storage[s]
}

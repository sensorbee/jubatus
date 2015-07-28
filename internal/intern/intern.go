package intern

// Intern is a mapping from strings to ints.
type Intern struct {
	storage map[string]int
	gen     int
}

// New creates a new Intern instance.
func New() *Intern {
	return &Intern{
		storage: make(map[string]int),
		gen:     0,
	}
}

// GetOrZero returns an ID for a string if the string is already registered.
// If the string is not registered this method returns zero,
func (i *Intern) GetOrZero(s string) int {
	return i.storage[s]
}

// Get returns an ID for a string. If the string was not registered, this
// method registers the string and returns an ID. This method is idempotent.
func (i *Intern) Get(s string) int {
	if i.GetOrZero(s) == 0 {
		i.gen++
		i.storage[s] = i.gen
		return i.gen
	}
	return i.storage[s]
}

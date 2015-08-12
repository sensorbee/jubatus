package nearest

type Neighbor interface {
	SetRow(id ID, v FeatureVector)
	NeighborRowFromID(id ID, size int) []IDist
	NeighborRowFromFV(v FeatureVector, size int) []IDist
	GetAllRows() []ID
}

type FeatureElement struct {
	Dim   string
	Value float32
}
type FeatureVector []FeatureElement

type IDist struct {
	ID   ID
	Dist float32
}

type ID int64

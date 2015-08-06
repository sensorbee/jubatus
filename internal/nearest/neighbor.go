package nearest

import (
	"pfi/sensorbee/sensorbee/data"
)

type Neighbor interface {
	SetRow(id ID, v FeatureVector)
	NeighborRowFromID(id ID, size int) []IDist
	NeighborRowFromFV(v FeatureVector, size int) []IDist
	GetAllRows() []ID
}

type FeatureVector data.Map

type IDist struct {
	ID   ID
	Dist float32
}

type ID int64

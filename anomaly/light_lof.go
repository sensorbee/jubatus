package anomaly

import (
	"errors"
	"math"
	"pfi/sensorbee/jubatus/internal/nearest"
	"pfi/sensorbee/sensorbee/data"
)

type LightLOF struct {
	nn     nearest.Neighbor
	nnNum  int
	rnnNum int

	kdists []float32
	lrds   []float32

	idgen ID
}

func NewLightLOF(hashNum, nnNum, rnnNum int) *LightLOF {
	return &LightLOF{
		nn:     nearest.NewMinhash(hashNum),
		nnNum:  nnNum,
		rnnNum: rnnNum,
	}
}

func (l *LightLOF) Add(v FeatureVector) (id ID, score float32) {
	l.idgen++
	id = l.idgen

	l.kdists = append(l.kdists, -1)
	l.lrds = append(l.lrds, -1)

	l.setRow(id, v)

	score = l.CalcScore(v)
	return id, score
}

// Update does not exist for LightLOF.
// func (l *LightLOF) Update(id ID, v FeatureVector) float32

func (l *LightLOF) Overwrite(id ID, v FeatureVector) (score float32, err error) {
	if id > l.idgen {
		return 0, errors.New("TODO")
	}

	l.setRow(id, v)

	score = l.CalcScore(v)

	return
}

func (l *LightLOF) CalcScore(v FeatureVector) float32 {
	lrd, neighborLRDs := l.collectLRDs(v)
	return calcLOF(lrd, neighborLRDs)
}

func (l *LightLOF) GetAllRows() []ID {
	// TODO: implement
	return nil
}

func (l *LightLOF) Clear() {
	// TODO: implement
}

func (l *LightLOF) setRow(id ID, v FeatureVector) {
	nnID := nearest.ID(id)
	l.nn.SetRow(nnID, nearest.FeatureVector(v))

	neighbors := l.nn.NeighborRowFromID(nnID, l.rnnNum)
	neighborsAndThePoint := append(neighbors, nearest.IDist{nnID, 0})

	nestedNeighbors := map[ID][]nearest.IDist{}
	for i := range neighborsAndThePoint {
		nnID := neighborsAndThePoint[i].ID
		id := ID(nnID)
		nnResult := l.nn.NeighborRowFromID(nnID, l.nnNum)
		nestedNeighbors[id] = nnResult
		l.kdists[id] = nnResult[len(nnResult)-1].Dist
	}

	for i := range neighborsAndThePoint {
		nnID := neighborsAndThePoint[i].ID
		id := ID(nnID)
		nn := nestedNeighbors[id]
		var lrd float32 = 1
		if len(nn) > 0 {
			length := minInt(len(nn), l.nnNum)
			var sumReachablity float32
			for i := 0; i < length; i++ {
				sumReachablity += maxFloat32(nn[i].Dist, l.kdists[nn[i].ID])
			}
			if sumReachablity == 0 {
				lrd = inf32
			} else {
				lrd = float32(length) / sumReachablity
			}
		}
		l.lrds[id] = lrd
	}
}

func (l *LightLOF) collectLRDs(v FeatureVector) (float32, []float32) {
	neighbors := l.nn.NeighborRowFromFV(nearest.FeatureVector(v), l.nnNum)
	if len(neighbors) == 0 {
		return inf32, nil
	}

	neighborLRDs := make([]float32, len(neighbors))
	parameters := make([]parameter, len(neighbors))

	for i := range neighbors {
		id := ID(neighbors[i].ID)
		p := l.getRowParameter(id)
		neighborLRDs[i] = p.lrd
		parameters[i] = p
	}

	var sumReachability float32
	for i := range neighbors {
		sumReachability += maxFloat32(neighbors[i].Dist, parameters[i].kdist)
	}

	if sumReachability == 0 {
		return inf32, neighborLRDs
	}

	return float32(len(neighbors)) / sumReachability, neighborLRDs
}

func (l *LightLOF) getRowParameter(id ID) parameter {
	return parameter{
		kdist: l.kdists[id],
		lrd:   l.lrds[id],
	}
}

func calcLOF(lrd float32, neighborLRDs []float32) float32 {
	if len(neighborLRDs) == 0 {
		if lrd == 0 {
			return 1
		}
		return inf32
	}

	var sum float32
	for _, x := range neighborLRDs {
		sum += x
	}
	if isInf32(sum) && isInf32(lrd) {
		return 1
	}

	return sum / (float32(len(neighborLRDs)) * lrd)
}

type FeatureVector data.Map

type ID int64

func minInt(x, y int) int {
	if x < y {
		return x
	}
	return y
}

func maxFloat32(x, y float32) float32 {
	if x < y {
		return y
	}
	return x
}

func isInf32(x float32) bool {
	return math.IsInf(float64(x), 0)
}

var inf32 = float32(math.Inf(1))

type parameter struct {
	kdist float32
	lrd   float32
}

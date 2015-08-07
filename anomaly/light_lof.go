package anomaly

import (
	"errors"
	"fmt"
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

	if len(l.kdists) < int(id) {
		l.extend(int(id))
	}

	l.setRow(id, v)

	score = l.CalcScore(v)
	return id, score
}

// Update does not exist for LightLOF.
// func (l *LightLOF) Update(id ID, v FeatureVector) float32

func (l *LightLOF) Overwrite(id ID, v FeatureVector) (score float32, err error) {
	if id <= 0 {
		return 0, fmt.Errorf("invalid id %d", id)
	}
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
		l.kdists[id-1] = nnResult[len(nnResult)-1].Dist
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
				sumReachablity += maxFloat32(nn[i].Dist, l.kdists[nn[i].ID-1])
			}
			if sumReachablity == 0 {
				lrd = inf32
			} else {
				lrd = float32(length) / sumReachablity
			}
		}
		l.lrds[id-1] = lrd
	}
}

func (l *LightLOF) collectLRDs(v FeatureVector) (float32, []float32) {
	neighbors := l.nn.NeighborRowFromFV(nearest.FeatureVector(v), l.nnNum)
	if len(neighbors) == 0 {
		return inf32, nil
	}

	neighborKDists := make([]float32, len(neighbors))
	neighborLRDs := make([]float32, len(neighbors))

	for i := range neighbors {
		id := ID(neighbors[i].ID)
		neighborKDists[i] = l.kdists[id-1]
		neighborLRDs[i] = l.lrds[id-1]
	}

	var sumReachability float32
	for i := range neighbors {
		sumReachability += maxFloat32(neighbors[i].Dist, neighborKDists[i])
	}

	if sumReachability == 0 {
		return inf32, neighborLRDs
	}

	return float32(len(neighbors)) / sumReachability, neighborLRDs
}

func (l *LightLOF) extend(n int) {
	n = maxInt(2*len(l.kdists), n)
	l.kdists = realloc(l.kdists, n)
	l.lrds = realloc(l.lrds, n)
}

func realloc(s []float32, n int) []float32 {
	ret := make([]float32, n)
	copy(ret, s)
	return ret
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

func maxInt(x, y int) int {
	if x < y {
		return y
	}
	return x
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

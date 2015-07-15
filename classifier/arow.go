package classifier

import (
	"errors"
)

type Arow struct {
	storage
	regWeight float64
}

func NewArow(regWeight float64) (*Arow, error) {
	if regWeight <= 0 {
		return nil, errors.New("regularization weight must be larger than zero.")
	}
	return &Arow{
		storage:   make(storage),
		regWeight: regWeight,
	}, nil
}

func (a *Arow) Train(v FeatureVector, label Label) error {
	if label == "" {
		return errors.New("label must not be empty.")
	}

	margin, variance, incorrectLabel := a.storage.calcMarginAndVarianceAndIncorrectLabel(v, label)

	if _, ok := a.storage[label]; !ok {
		a.storage[label] = make(W)
	}

	if margin <= -1 {
		return nil
	}

	var beta float64 = 1 / (variance + 1 / a.regWeight)
	var alpha float64 = (1 + margin) * beta

	for _, elem := range v {
		negVal := [2]float64{0, 1}
		if val, ok := a.storage[incorrectLabel][elem.Dim]; ok {
			copy(negVal[:], val[:2])
		}
		posVal := [2]float64{0, 1}
		if val, ok := a.storage[label][elem.Dim]; ok {
			copy(posVal[:], val[:2])
		}

		if incorrectLabel != "" {
			a.storage[incorrectLabel][elem.Dim] = [2]float64{
				negVal[0] - alpha * negVal[1] * elem.Value,
				negVal[1] - beta * negVal[1] * negVal[1] * elem.Value * elem.Value,
			}
		}

		a.storage[label][elem.Dim] = [2]float64{
			posVal[0] + alpha * posVal[1] * elem.Value,
			posVal[1] - beta * posVal[1] * posVal[1] * elem.Value * elem.Value,
		}
	}

	return nil
}

func (a *Arow) Classify(v FeatureVector) Label {
	scores := a.storage.calcScores(v)
	_, ls, _ := scores.MinMax()
	return ls.Label
}

func (a *Arow) Clear() {
	a.storage = make(storage)
}

type Dim string
type FeatureElement struct {
	Dim
	Value float64
}
type FeatureVector []FeatureElement

type Label string
type W map[Dim][2]float64
type storage map[Label]W

type LScore struct {
	Label Label
	Score float64
}
type LScores []LScore

func (s LScores) MinMax() (min *LScore, max *LScore, err error) {
	if len(s) == 0 {
		err = errors.New("TODO")
		return
	}
	min = &s[0]
	max = &s[0]
	for i, ls := range s[1:] {
		score := ls.Score
		if min.Score > score {
			min = &s[i+1]
		} else if max.Score < score {
			max = &s[i+1]
		}
	}
	return
}

func (s LScores) Find(l Label) int {
	for i, ls := range s {
		if ls.Label == l {
			return i
		}
	}
	return -1
}

// jubatus::core::classifier::linear_classifier::classify_with_scores
func (s storage) calcScores(v FeatureVector) LScores {
	scores := make(LScores, 0, len(s))
	for l, w := range s {
		ls := LScore{Label: l}
		for _, x := range v {
			ls.Score += x.Value * w[x.Dim][0]
		}
		scores = append(scores, ls)
	}
	return scores
}

func (s storage) calcMarginAndVarianceAndIncorrectLabel(v FeatureVector, l Label) (margin float64, variance float64, incorrect Label) {
	if len(s) == 0 {
		return
	}

	scores := s.calcScores(v)
	corrIx := scores.Find(l)
	if corrIx < 0 {
		_, incorr, _ := scores.MinMax()
		margin = incorr.Score
		incorrect = incorr.Label
		incorrV := s[incorrect]
		for _, elem := range v {
			if _, ok := incorrV[elem.Dim]; ok {
				variance += (1 + incorrV[elem.Dim][1]) * elem.Value
			} else {
				variance += 2 * elem.Value
			}
		}
		return
	}
	corr := &scores[corrIx]
	if len(s) == 1 {
		margin = -corr.Score
		corrV := s[l]
		for _, elem := range v {
			if _, ok := corrV[elem.Dim]; ok {
				variance += (corrV[elem.Dim][1] + 1) * elem.Value
			} else {
				variance += 2 * elem.Value
			}
		}
	} else {
		scores[0], scores[corrIx] = scores[corrIx], scores[0]
		_, incorr, _ := scores[1:].MinMax()
		margin = incorr.Score - corr.Score
		corrV := s[l]
		incorrect = incorr.Label
		incorrV := s[incorrect]
		for _, elem := range v {
			c := 1.0
			i := 1.0
			if _, ok := corrV[elem.Dim]; ok {
				c = corrV[elem.Dim][1]
			}
			if _, ok := incorrV[elem.Dim]; ok {
				i = incorrV[elem.Dim][1]
			}
			variance += (c + i) * elem.Value
		}
	}
	return
}

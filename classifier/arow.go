package classifier

import (
	"errors"
	"sort"
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

	if _, ok := a.storage[label]; !ok {
		a.storage[label] = make(W)
	}

	scores := a.storage.calcScores(v)
	corr, incorr := scores.getCorrectAndIncorrect(label)
	margin := calcMargin(corr, incorr)

	if margin <= -1 {
		return nil
	}

	variance := calcVariance(v, a.storage[label], a.storage[incorr.labelOrElse("")])

	var beta float64 = 1 / (variance + 1/a.regWeight)
	var alpha float64 = (1 + margin) * beta

	for _, elem := range v {
		dim := elem.Dim
		value := elem.Value

		negVal := [2]float64{0, 1}
		if incorr != nil {
			if val, ok := a.storage[incorr.Label][dim]; ok {
				copy(negVal[:], val[:2])
			}
		}
		posVal := [2]float64{0, 1}
		if val, ok := a.storage[label][dim]; ok {
			copy(posVal[:], val[:2])
		}

		if incorr != nil {
			a.storage[incorr.Label][dim] = [2]float64{
				negVal[0] - alpha*negVal[1]*value,
				negVal[1] - beta*negVal[1]*negVal[1]*value*value,
			}
		}

		a.storage[label][dim] = [2]float64{
			posVal[0] + alpha*posVal[1]*value,
			posVal[1] - beta*posVal[1]*posVal[1]*value*value,
		}
	}

	return nil
}

func (a *Arow) Classify(v FeatureVector) LScores {
	scores := a.storage.calcScores(v)
	sort.Sort(lScores(scores))
	return scores
}

func (a *Arow) Clear() {
	a.storage = make(storage)
}

func (a *Arow) RegWeight() float64 {
	return a.regWeight
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

func (ls *LScore) labelOrElse(defaultL Label) Label {
	if ls == nil {
		return defaultL
	}
	return ls.Label
}

func (ls *LScore) scoreOrElse(defaultS float64) float64 {
	if ls == nil {
		return defaultS
	}
	return ls.Score
}

type LScores []LScore

func (s LScores) Max() *LScore {
	if len(s) == 0 {
		return nil
	}
	ret := &s[0]
	for i := 1; i < len(s); i++ {
		if s[i].Score > ret.Score {
			ret = &s[i]
		}
	}
	return ret
}

func (s LScores) maxExcept(except int) *LScore {
	if except < 0 || except >= len(s) {
		return s.Max()
	}

	l := s[:except].Max()
	r := s[except+1:].Max()
	if l == nil {
		return r
	}
	if r == nil {
		return l
	}
	if l.Score < r.Score {
		return r
	}
	return l
}

func (s LScores) Find(l Label) int {
	for i, ls := range s {
		if ls.Label == l {
			return i
		}
	}
	return -1
}

func (s LScores) getCorrectAndIncorrect(l Label) (correct *LScore, incorrect *LScore) {
	corrIx := s.Find(l)
	if corrIx >= 0 {
		correct = &s[corrIx]
		incorrect = s.maxExcept(corrIx)
	} else {
		incorrect = s.Max()
	}

	return
}

type lScores LScores

func (s lScores) Len() int {
	return len(s)
}

func (s lScores) Less(i, j int) bool {
	return s[i].Score > s[j].Score
}

func (s lScores) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
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

func calcMargin(correct *LScore, incorrect *LScore) float64 {
	return incorrect.scoreOrElse(0) - correct.scoreOrElse(0)
}

func calcVariance(v FeatureVector, w1, w2 W) float64 {
	variance := 0.0
	for _, elem := range v {
		dim := elem.Dim
		val := elem.Value
		variance += (w1.covar(dim) + w2.covar(dim)) * val * val
	}
	return variance
}

// TODO: consider to rename
func (w W) covar(dim Dim) float64 {
	if w == nil {
		return 1
	}
	if c, ok := w[dim]; ok {
		return c[1]
	}
	return 1
}

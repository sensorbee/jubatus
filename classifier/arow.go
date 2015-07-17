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
		a.storage[label] = make(weights)
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

	var incorrWeights weights
	if incorr != nil {
		incorrWeights = a.storage[incorr.Label]
	}
	var corrWeights weights = a.storage[label]

	for _, elem := range v {
		dim := elem.Dim
		value := elem.Value

		if incorr != nil {
			incorrWeights.negativeUpdate(alpha, beta, dim, value)
		}

		corrWeights.positiveUpdate(alpha, beta, dim, value)
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
type weight struct {
	weight     float64
	confidence float64
}
type weights map[Dim]weight
type storage map[Label]weights

func initialWeight() weight {
	return weight{
		weight:     0,
		confidence: 1,
	}
}

func (ws weights) negativeUpdate(alpha, beta float64, dim Dim, x float64) {
	ws.update(alpha, beta, dim, x, (*weight).negativeUpdate)
}

func (ws weights) positiveUpdate(alpha, beta float64, dim Dim, x float64) {
	ws.update(alpha, beta, dim, x, (*weight).positiveUpdate)
}

func (ws weights) update(alpha, beta float64, dim Dim, x float64, f weightUpdateFunction) {
	var weight weight
	if w, ok := ws[dim]; ok {
		weight = w
	} else {
		weight = initialWeight()
	}
	f(&weight, alpha, beta, x)
	ws[dim] = weight
}

type weightUpdateFunction func(w *weight, alpha, beta, x float64)

func (w *weight) negativeUpdate(alpha, beta, x float64) {
	aTerm := w.calcAlphaTerm(alpha, x)
	bTerm := w.calcBetaTerm(beta, x)
	w.weight -= aTerm
	w.confidence -= bTerm
	return
}

func (w *weight) positiveUpdate(alpha, beta, x float64) {
	aTerm := w.calcAlphaTerm(alpha, x)
	bTerm := w.calcBetaTerm(beta, x)
	w.weight += aTerm
	w.confidence -= bTerm
	return
}

func (w *weight) calcAlphaTerm(alpha, x float64) float64 {
	return alpha * w.confidence * x
}

func (w *weight) calcBetaTerm(beta, x float64) float64 {
	conf := w.confidence
	return beta * conf * conf * x * x
}

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

func (s LScores) max() *LScore {
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

func (s LScores) maxExcept(exceptIx int) *LScore {
	if exceptIx < 0 || exceptIx >= len(s) {
		return s.max()
	}

	l := s[:exceptIx].max()
	r := s[exceptIx+1:].max()
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

func (s LScores) find(l Label) int {
	for i, ls := range s {
		if ls.Label == l {
			return i
		}
	}
	return -1
}

func (s LScores) getCorrectAndIncorrect(l Label) (correct *LScore, incorrect *LScore) {
	corrIx := s.find(l)
	if corrIx >= 0 {
		correct = &s[corrIx]
		incorrect = s.maxExcept(corrIx)
	} else {
		incorrect = s.max()
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
			ls.Score += x.Value * w[x.Dim].weight
		}
		scores = append(scores, ls)
	}
	return scores
}

func calcMargin(correct *LScore, incorrect *LScore) float64 {
	return incorrect.scoreOrElse(0) - correct.scoreOrElse(0)
}

func calcVariance(v FeatureVector, w1, w2 weights) float64 {
	variance := 0.0
	for _, elem := range v {
		dim := elem.Dim
		val := elem.Value
		variance += (w1.covar(dim) + w2.covar(dim)) * val * val
	}
	return variance
}

// TODO: consider to rename
func (ws weights) covar(dim Dim) float64 {
	if ws == nil {
		return 1
	}
	if w, ok := ws[dim]; ok {
		return w.confidence
	}
	return 1
}

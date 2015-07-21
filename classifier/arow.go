package classifier

import (
	"errors"
	"sort"
	"sync"
)

type AROW struct {
	model
	regWeight float32
	m         sync.Mutex
}

func NewArow(regWeight float32) (*AROW, error) {
	if regWeight <= 0 {
		return nil, errors.New("regularization weight must be larger than zero.")
	}
	return &AROW{
		model:     make(model),
		regWeight: regWeight,
	}, nil
}

func (a *AROW) Train(v FeatureVector, label Label) error {
	a.m.Lock()
	defer a.m.Unlock()

	if label == "" {
		return errors.New("label must not be empty.")
	}

	if _, ok := a.model[label]; !ok {
		a.model[label] = make(weights)
	}

	scores := a.model.scores(v)
	corr, incorr := scores.correctAndIncorrect(label)
	margin := margin(corr, incorr)

	if margin <= -1 {
		return nil
	}

	variance := variance(v, a.model[label], a.model[incorr.labelOrElse("")])

	var beta float32 = 1 / (variance + 1/a.regWeight)
	var alpha float32 = (1 + margin) * beta

	var incorrWeights weights
	if incorr != nil {
		incorrWeights = a.model[incorr.Label]
	}
	var corrWeights weights = a.model[label]

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

func (a *AROW) Classify(v FeatureVector) LScores {
	a.m.Lock()
	defer a.m.Unlock()
	scores := a.model.scores(v)
	sort.Sort(lScores(scores))
	return scores
}

func (a *AROW) Clear() {
	a.m.Lock()
	defer a.m.Unlock()
	a.model = make(model)
}

func (a *AROW) RegWeight() float32 {
	return a.regWeight
}

type Dim string
type FeatureElement struct {
	Dim
	Value float32
}
type FeatureVector []FeatureElement

type Label string
type weight struct {
	weight     float32
	covariance float32
}
type weights map[Dim]weight
type model map[Label]weights

func initialWeight() weight {
	return weight{
		weight:     0,
		covariance: 1,
	}
}

func (ws weights) negativeUpdate(alpha, beta float32, dim Dim, x float32) {
	ws.update(alpha, beta, dim, x, (*weight).negativeUpdate)
}

func (ws weights) positiveUpdate(alpha, beta float32, dim Dim, x float32) {
	ws.update(alpha, beta, dim, x, (*weight).positiveUpdate)
}

func (ws weights) update(alpha, beta float32, dim Dim, x float32, f weightUpdateFunction) {
	var weight weight
	if w, ok := ws[dim]; ok {
		weight = w
	} else {
		weight = initialWeight()
	}
	f(&weight, alpha, beta, x)
	ws[dim] = weight
}

type weightUpdateFunction func(w *weight, alpha, beta, x float32)

func (w *weight) negativeUpdate(alpha, beta, x float32) {
	aTerm := w.alphaTerm(alpha, x)
	bTerm := w.betaTerm(beta, x)
	w.weight -= aTerm
	w.covariance -= bTerm
	return
}

func (w *weight) positiveUpdate(alpha, beta, x float32) {
	aTerm := w.alphaTerm(alpha, x)
	bTerm := w.betaTerm(beta, x)
	w.weight += aTerm
	w.covariance -= bTerm
	return
}

func (w *weight) alphaTerm(alpha, x float32) float32 {
	return alpha * w.covariance * x
}

func (w *weight) betaTerm(beta, x float32) float32 {
	conf := w.covariance
	return beta * conf * conf * x * x
}

type LScore struct {
	Label Label
	Score float32
}

func (ls *LScore) labelOrElse(defaultL Label) Label {
	if ls == nil {
		return defaultL
	}
	return ls.Label
}

func (ls *LScore) scoreOrElse(defaultS float32) float32 {
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

func (s LScores) correctAndIncorrect(l Label) (correct *LScore, incorrect *LScore) {
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
func (s model) scores(v FeatureVector) LScores {
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

func margin(correct *LScore, incorrect *LScore) float32 {
	return incorrect.scoreOrElse(0) - correct.scoreOrElse(0)
}

func variance(v FeatureVector, w1, w2 weights) float32 {
	var variance float32 = 0
	for _, elem := range v {
		dim := elem.Dim
		val := elem.Value
		variance += (w1.covariance(dim) + w2.covariance(dim)) * val * val
	}
	return variance
}

// TODO: consider to rename
func (ws weights) covariance(dim Dim) float32 {
	if ws == nil {
		return 1
	}
	if w, ok := ws[dim]; ok {
		return w.covariance
	}
	return 1
}

package classifier

import (
	"errors"
	"pfi/sensorbee/jubatus/common/intern"
	"pfi/sensorbee/sensorbee/data"
	"sync"
)

// AROW holds a model for classification.
type AROW struct {
	model
	intern *intern.Intern
	m      sync.RWMutex

	regWeight float32
}

// NewAROW creates an AROW model. regWeight means sensitivity for data. When regWeight is large,
// the model learns quickly but harms from noise. regWeight must be larger than zero.
func NewAROW(regWeight float32) (*AROW, error) {
	if regWeight <= 0 {
		return nil, errors.New("regularization weight must be larger than zero.")
	}
	return &AROW{
		model:     make(model),
		regWeight: regWeight,
		intern:    intern.New(),
	}, nil
}

// Train trains a model with a feature vector and a label.
func (a *AROW) Train(v FeatureVector, label Label) error {
	if label == "" {
		return errors.New("label must not be empty.")
	}

	a.m.Lock()
	defer a.m.Unlock()

	if _, ok := a.model[label]; !ok {
		a.model[label] = make(weights)
	}

	fvForScores, fvFull, err := v.toInternal(a.intern)
	if err != nil {
		return err
	}
	scores := a.model.scores(fvForScores)
	corr, incorr := scores.correctAndIncorrect(label)
	margin := margin(corr, incorr)

	if margin <= -1 {
		return nil
	}

	variance := variance(fvFull, a.model[label], a.model[incorr.labelOrElse("")])

	var beta float32 = 1 / (variance + 1/a.regWeight)
	var alpha float32 = (1 + margin) * beta

	var incorrWeights weights
	if incorr != nil {
		incorrWeights = a.model[incorr.Label]
	}
	var corrWeights weights = a.model[label]

	for _, elem := range fvFull {
		dim := elem.dim
		value := elem.value

		if incorr != nil {
			incorrWeights.negativeUpdate(alpha, beta, dim, value)
		}

		corrWeights.positiveUpdate(alpha, beta, dim, value)
	}

	return nil
}

// Classify classifies a feature vector. This function returns
// all labels and scores.
func (a *AROW) Classify(v FeatureVector) (LScores, error) {
	a.m.RLock()
	defer a.m.RUnlock()
	intfv, err := v.toInternalForScores(a.intern)
	if err != nil {
		return nil, err
	}
	scores := a.model.scores(intfv)
	return scores, nil
}

// Clear clears a model.
func (a *AROW) Clear() {
	a.m.Lock()
	defer a.m.Unlock()
	a.model = make(model)
	a.intern = intern.New()
}

// RegWeight returns regularization weight.
func (a *AROW) RegWeight() float32 {
	return a.regWeight
}

// FeatureVector is a type for feature vectors.
type FeatureVector data.Map

func (v FeatureVector) toInternalForScores(intern *intern.Intern) (fVectorForScores, error) {
	ret := make(fVectorForScores, 0, len(v))
	for f, val := range v {
		xx, err := data.ToFloat(val)
		if err != nil {
			return nil, err
		}
		x := float32(xx)
		if d := intern.MayGet(f); d != 0 {
			ret = append(ret, fElement{dim(d), x})
		}
	}
	return ret, nil
}

func (v FeatureVector) toInternal(intern *intern.Intern) (fVectorForScores, fVector, error) {
	full := make(fVector, len(v))
	l, r := 0, len(v)
	for f, val := range v {
		xx, err := data.ToFloat(val)
		if err != nil {
			return nil, nil, err
		}
		x := float32(xx)
		if d := intern.MayGet(f); d != 0 {
			full[l] = fElement{dim(d), x}
			l++
		} else {
			r--
			full[r] = fElement{dim(intern.Get(f)), x}
		}
	}
	return fVectorForScores(full[:l]), full, nil
}

type dim int
type fElement struct {
	dim
	value float32
}
type fVector []fElement
type fVectorForScores []fElement

// Label represents labels for classification.
type Label string
type weight struct {
	weight     float32
	covariance float32
}
type weights map[dim]weight
type model map[Label]weights

func initialWeight() weight {
	return weight{
		weight:     0,
		covariance: 1,
	}
}

func (ws weights) negativeUpdate(alpha, beta float32, dim dim, x float32) {
	ws.update(alpha, beta, dim, x, (*weight).negativeUpdate)
}

func (ws weights) positiveUpdate(alpha, beta float32, dim dim, x float32) {
	ws.update(alpha, beta, dim, x, (*weight).positiveUpdate)
}

func (ws weights) update(alpha, beta float32, dim dim, x float32, f weightUpdateFunction) {
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

// LScore is a pair of a label and a score.
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

// LScores is pairs of a label and a score.
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

// jubatus::core::classifier::linear_classifier::classify_with_scores
func (s model) scores(v fVectorForScores) LScores {
	scores := make(LScores, 0, len(s))
	for l, w := range s {
		ls := LScore{Label: l}
		for _, x := range v {
			ls.Score += x.value * w[x.dim].weight
		}
		scores = append(scores, ls)
	}
	return scores
}

func margin(correct *LScore, incorrect *LScore) float32 {
	return incorrect.scoreOrElse(0) - correct.scoreOrElse(0)
}

func variance(v fVector, w1, w2 weights) float32 {
	var variance float32 = 0
	for _, elem := range v {
		dim := elem.dim
		val := elem.value
		variance += (w1.covariance(dim) + w2.covariance(dim)) * val * val
	}
	return variance
}

// TODO: consider to rename
func (ws weights) covariance(dim dim) float32 {
	if ws == nil {
		return 1
	}
	if w, ok := ws[dim]; ok {
		return w.covariance
	}
	return 1
}

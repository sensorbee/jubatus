package classifier

import (
	"errors"
	"fmt"
	"github.com/ugorji/go/codec"
	"io"
	"math"
	"pfi/sensorbee/jubatus/internal/intern"
	"pfi/sensorbee/jubatus/internal/nested"
	"pfi/sensorbee/sensorbee/data"
	"sync"
)

// AROW holds a model for classification.
type AROW struct {
	model  model
	intern *intern.Intern
	m      sync.RWMutex

	regWeight float32
}

// NewAROW creates an AROW model. regWeight means sensitivity for data. When regWeight is large,
// the model learns quickly but harms from noise. regWeight must be larger than zero.
func NewAROW(regWeight float32) (*AROW, error) {
	if regWeight <= 0 {
		return nil, errors.New("regularization weight must be larger than zero")
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
		return errors.New("label must not be empty")
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
	incorr, _ := scores.maxExcept(label)
	margin := scores.margin(label, incorr)

	if margin <= -1 {
		return nil
	}

	variance := variance(fvFull, a.model[label], a.model[incorr])

	beta := 1 / (variance + 1/a.regWeight)
	alpha := (1 + margin) * beta

	var incorrWeights weights
	if incorr != "" {
		incorrWeights = a.model[incorr]
	}
	corrWeights := a.model[label]

	for _, elem := range fvFull {
		dim := elem.dim
		value := elem.value

		if incorr != "" {
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

var (
	arowFormatVersion uint8 = 1
)

type arowMsgpack struct {
	_struct   struct{} `codec:",toarray"`
	Model     model
	RegWeight float32
}

// Save saves the current state of AROW.
func (a *AROW) Save(w io.Writer) error {
	a.m.RLock()
	defer a.m.RUnlock()

	if _, err := w.Write([]byte{arowFormatVersion}); err != nil {
		return err
	}

	enc := codec.NewEncoder(w, classifierMsgpackHandle)
	if err := enc.Encode(&arowMsgpack{
		Model:     a.model,
		RegWeight: a.regWeight,
	}); err != nil {
		return err
	}
	return a.intern.Save(w)
}

// TODO: Provide Load which is memory&CPU efficient than the current
// swapping style load.

// LoadAROW loads AROW from the saved data.
func LoadAROW(r io.Reader) (*AROW, error) {
	formatVersion := make([]byte, 1)
	if _, err := r.Read(formatVersion); err != nil {
		return nil, err
	}

	switch formatVersion[0] {
	case 1:
		return loadAROWFormatV1(r)
	default:
		return nil, fmt.Errorf("unsupported format version of AROW container: %v", formatVersion[0])
	}
}

func loadAROWFormatV1(r io.Reader) (*AROW, error) {
	m := arowMsgpack{}
	dec := codec.NewDecoder(r, classifierMsgpackHandle)
	if err := dec.Decode(&m); err != nil {
		return nil, err
	}
	i, err := intern.Load(r)
	if err != nil {
		return nil, err
	}

	return &AROW{
		model:     m.Model,
		intern:    i,
		regWeight: m.RegWeight,
	}, nil
}

// RegWeight returns regularization weight.
func (a *AROW) RegWeight() float32 {
	return a.regWeight
}

// FeatureVector is a type for feature vectors.
type FeatureVector data.Map

// toInternalForScores converts a feature vector to internal format. It requires read lock for intern.
func (v FeatureVector) toInternalForScores(intern *intern.Intern) (fVectorForScores, error) {
	ret := make(fVectorForScores, 0, len(v))
	err := nested.Flatten(data.Map(v), func(key string, value float32) {
		if d := intern.GetOrZero(key); d != 0 {
			ret = append(ret, fElement{dim(d), value})
		}
	})
	if err != nil {
		return nil, err
	}
	return ret, nil
}

// toInternal converts a feature vector to internal format. It requires write lock for intern.
func (v FeatureVector) toInternal(intern *intern.Intern) (fVectorForScores, fVector, error) {
	full := make(fVector, 0, len(v))
	err := nested.Flatten(data.Map(v), func(key string, value float32) {
		full = append(full, fElement{dim(intern.Get(key)), value})
	})
	if err != nil {
		return nil, nil, err
	}
	return fVectorForScores(full), full, nil
}

type appender func(string, float32)

func toInternalImpl(keyPrefix string, v data.Map, ap appender) error {
	for f, val := range v {
		if m, err := data.AsMap(val); err == nil {
			err := toInternalImpl(fmt.Sprint(keyPrefix, f, "\x00"), m, ap)
			if err != nil {
				return err
			}
		} else {
			xx, err := data.ToFloat(val)
			if err != nil {
				// TODO: return better error
				return err
			}
			x := float32(xx)
			ap(keyPrefix+f, x)
		}
	}
	return nil
}

type dim int
type fElement struct {
	dim   dim
	value float32
}
type fVector []fElement
type fVectorForScores []fElement

// Label represents labels for classification.
type Label string
type weight struct {
	_struct    struct{} `codec:",toarray"`
	Weight     float32
	Covariance float32
}
type weights map[dim]weight
type model map[Label]weights

func initialWeight() weight {
	return weight{
		Weight:     0,
		Covariance: 1,
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
	w.Weight -= aTerm
	w.Covariance -= bTerm
	return
}

func (w *weight) positiveUpdate(alpha, beta, x float32) {
	aTerm := w.alphaTerm(alpha, x)
	bTerm := w.betaTerm(beta, x)
	w.Weight += aTerm
	w.Covariance -= bTerm
	return
}

func (w *weight) alphaTerm(alpha, x float32) float32 {
	return alpha * w.Covariance * x
}

func (w *weight) betaTerm(beta, x float32) float32 {
	conf := w.Covariance
	return beta * conf * conf * x * x
}

// LScores is a type representing a map from labels to scores.
type LScores data.Map

func (s LScores) score(l Label) float32 {
	vsc, ok := s[string(l)]
	if !ok {
		return 0
	}
	sc, _ := data.AsFloat(vsc)
	return float32(sc)
}

// Max returns a max scored label and the score
func (s LScores) Max() (label Label, score float32) {
	return s.maxExcept("")
}

func (s LScores) maxExcept(except Label) (label Label, score float32) {
	score = float32(math.Inf(-1))
	for l := range s {
		la := Label(l)
		sc := s.score(la)
		if sc > score && la != except {
			label = la
			score = sc
		}
	}
	return
}

func (s LScores) margin(correct Label, incorrect Label) float32 {
	return s.score(incorrect) - s.score(correct)
}

// jubatus::core::classifier::linear_classifier::classify_with_scores
func (s model) scores(v fVectorForScores) LScores {
	scores := make(LScores)
	for l, w := range s {
		var score float32
		for _, x := range v {
			score += x.value * w[x.dim].Weight
		}
		scores[string(l)] = data.Float(score)
	}
	return scores
}

func variance(v fVector, w1, w2 weights) float32 {
	var variance float32
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
		return w.Covariance
	}
	return 1
}

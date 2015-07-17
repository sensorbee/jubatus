package classifier

import (
	"fmt"
	"math/rand"
)


type shogun struct {
	family string
	given  string
}

func unigram(s string) FeatureVector {
	fv := FeatureVector{}
	for _, r := range s {
		fv = append(fv, FeatureElement{
			Dim:   Dim(r),
			Value: 1,
		})
	}
	return fv
}

func Example() {
	shogunList := []shogun{
		{"徳川", "家康"}, {"徳川", "秀忠"}, {"徳川", "家光"},	{"徳川", "家綱"},
		{"徳川", "綱吉"}, {"徳川", "家宣"}, {"徳川", "家継"},	{"徳川", "吉宗"},
		{"徳川", "家重"}, {"徳川", "家治"}, {"徳川", "家斉"}, {"徳川", "家慶"},
		{"徳川", "家定"}, {"徳川", "家茂"},

		{"足利", "尊氏"}, {"足利", "義詮"}, {"足利", "義満"}, {"足利", "義持"},
		{"足利", "義量"}, {"足利", "義教"}, {"足利", "義勝"}, {"足利", "義政"},
		{"足利", "義尚"}, {"足利", "義稙"}, {"足利", "義澄"}, {"足利", "義稙"},
		{"足利", "義晴"}, {"足利", "義輝"}, {"足利", "義栄"},

		{"北条", "時政"}, {"北条", "義時"}, {"北条", "泰時"}, {"北条", "経時"},
		{"北条", "時頼"}, {"北条", "長時"}, {"北条", "政村"}, {"北条", "時宗"},
		{"北条", "貞時"}, {"北条", "師時"}, {"北条", "宗宣"}, {"北条", "煕時"},
		{"北条", "基時"}, {"北条", "高時"}, {"北条", "貞顕"},
	}

	shuffledShogunList := make([]shogun, len(shogunList))
	perm := rand.Perm(len(shogunList))
	for i, v := range perm {
		shuffledShogunList[v] = shogunList[i]
	}

	var arow, _ = NewArow(1)
	for _, s := range shuffledShogunList {
		fv := unigram(s.given)
		arow.Train(fv, Label(s.family))
	}

	scores := arow.Classify(unigram("慶喜"))
	fmt.Println(scores[0].Label)
	scores = arow.Classify(unigram("義昭"))
	fmt.Println(scores[0].Label)
	scores = arow.Classify(unigram("守時"))
	fmt.Println(scores[0].Label)

	// Output:
	// 徳川
	// 足利
	// 北条
}

package main

import (
	"fmt"
	"math/rand"
	"pfi/sensorbee/jubatus/classifier"
)

var arow, _ = classifier.NewArow(1)

func main() {
	shuffledShogunList := make([]shogun, len(shogunList))
	perm := rand.Perm(len(shogunList))
	for i, v := range perm {
		shuffledShogunList[v] = shogunList[i]
	}

	for _, s := range shuffledShogunList {
		fv := classifier.FeatureVector{}
		for _, r := range s.given {
			fv = append(fv, classifier.FeatureElement{
				Dim:   classifier.Dim(r),
				Value: 1,
			})
		}
		arow.Train(fv, classifier.Label(s.family))
	}

	classifyAndPrintln("慶喜")
	classifyAndPrintln("義昭")
	classifyAndPrintln("守時")
}

func classifyAndPrintln(given string) {
	fv := classifier.FeatureVector{}
	for _, c := range given {
		fv = append(fv, classifier.FeatureElement{
			Dim:   classifier.Dim(c),
			Value: 1,
		})
	}
	label := arow.Classify(fv)
	fmt.Println(given, label)
}

type shogun struct {
	family string
	given  string
}

var shogunList []shogun = []shogun{
	{"徳川", "家康"},
	{"徳川", "秀忠"},
	{"徳川", "家光"},
	{"徳川", "家綱"},
	{"徳川", "綱吉"},
	{"徳川", "家宣"},
	{"徳川", "家継"},
	{"徳川", "吉宗"},
	{"徳川", "家重"},
	{"徳川", "家治"},
	{"徳川", "家斉"},
	{"徳川", "家慶"},
	{"徳川", "家定"},
	{"徳川", "家茂"},

	{"足利", "尊氏"},
	{"足利", "義詮"},
	{"足利", "義満"},
	{"足利", "義持"},
	{"足利", "義量"},
	{"足利", "義教"},
	{"足利", "義勝"},
	{"足利", "義政"},
	{"足利", "義尚"},
	{"足利", "義稙"},
	{"足利", "義澄"},
	{"足利", "義稙"},
	{"足利", "義晴"},
	{"足利", "義輝"},
	{"足利", "義栄"},

	{"北条", "時政"},
	{"北条", "義時"},
	{"北条", "泰時"},
	{"北条", "経時"},
	{"北条", "時頼"},
	{"北条", "長時"},
	{"北条", "政村"},
	{"北条", "時宗"},
	{"北条", "貞時"},
	{"北条", "師時"},
	{"北条", "宗宣"},
	{"北条", "煕時"},
	{"北条", "基時"},
	{"北条", "高時"},
	{"北条", "貞顕"},
}

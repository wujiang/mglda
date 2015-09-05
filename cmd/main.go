package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"io/ioutil"
	"os"

	"gitlab.com/wujiang/mglda"
)

const (
	globalK        = 60
	localK         = 30
	gamma          = 0.1
	globalAlpha    = 0.1
	localAlpha     = 0.1
	globalAlphaMix = 0.1
	localAlphaMix  = 0.1
	globalBeta     = 0.1
	localBeta      = 0.1
	uT             = 3
)

type Data struct {
	Docs       []mglda.Document `json:"docs"`
	Vocabulary []string         `json:"vocabulary"`
}

type FakeData struct {
	Docs       []mglda.Document `json:"docs"`
	Vocabulary []string         `json:"vocabulary"`
}

func (d *Data) parse(fn string) error {
	bt, err := ioutil.ReadFile(fn)
	if err != nil {
		return err
	}
	err = json.Unmarshal(bt, d)
	return err
}

var (
	category = flag.String("category", "/home/wjiang/workspace/bitbucket/review/data/amazon/cellphone/corpus.json", "category")
)

func main() {
	data := Data{}
	if err := data.parse(*category); err != nil {
		panic(err)
	}
	uW := len(data.Vocabulary)
	docs := data.Docs
	m := mglda.NewMGLDA(globalK, localK, gamma, globalAlpha, localAlpha,
		globalAlphaMix, localAlphaMix, globalBeta, localBeta,
		uT, uW, &docs)
	out, _ := os.Create("woot")
	defer out.Close()
	wt := bufio.NewWriter(out)
	defer wt.Flush()
	mglda.Learning(m, 1000, data.Vocabulary, wt)
}

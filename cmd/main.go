package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"io/ioutil"
	"os"

	"gitlab.com/wujiang/mglda"
)

type Configuration struct {
	GlobalK        int     `json:"global_k"`
	LocalK         int     `json:"local_k"`
	Gamma          float64 `json:"gamma"`
	GlobalAlpha    float64 `json:"global_alpha"`
	LocalAlpha     float64 `json:"local_alpha"`
	GlobalAlphaMix float64 `json:"global_alpha_mix"`
	LocalAlphaMix  float64 `jons:"local_alpha_mix"`
	GlobalBeta     float64 `json:"global_beta"`
	LocalBeta      float64 `json:"local_beta"`
	T              int     `json:"t"`
	Interation     int     `josn:"interation"`
	DataPath       string  `json:"data_path"`
	OutPath        string  `json:"out_path"`
}

func (d *Configuration) parse(fn string) error {
	bt, err := ioutil.ReadFile(fn)
	if err != nil {
		return err
	}
	err = json.Unmarshal(bt, d)
	return err
}

type Data struct {
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
	confFile = flag.String("c", "conf.json", "Configuration file")
)

func main() {
	flag.Parse()
	conf := &Configuration{}
	if err := conf.parse(*confFile); err != nil {
		panic(err)
	}

	data := Data{}
	if err := data.parse(conf.DataPath); err != nil {
		panic(err)
	}
	uW := len(data.Vocabulary)
	docs := data.Docs
	m := mglda.NewMGLDA(conf.GlobalK, conf.LocalK, conf.Gamma,
		conf.GlobalAlpha, conf.LocalAlpha,
		conf.GlobalAlphaMix, conf.LocalAlphaMix,
		conf.GlobalBeta, conf.LocalBeta,
		conf.T, uW, &docs)
	out, _ := os.Create(conf.OutPath)
	defer out.Close()
	wt := bufio.NewWriter(out)
	defer wt.Flush()
	mglda.Learning(m, conf.Interation, data.Vocabulary, wt)
}

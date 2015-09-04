package main

import (
	"encoding/json"
	"io/ioutil"
)

type Data struct {
	Docs       []Document `json:"docs"`
	Vocabulary []string   `json:"vocabulary"`
}

func (d *Data) parse(fn string) error {
	bt, err := ioutil.ReadFile(fn)
	if err != nil {
		return err
	}
	err = json.Unmarshal(bt, d)
	return err
}

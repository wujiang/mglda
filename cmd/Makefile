.PHONY: build doc local

default: build

build: mglda
	GOOS=linux GOARCH=amd64 go build -o mglda

doc:
	godoc -http=:8765

local:
	go build -o mglda

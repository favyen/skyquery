package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"path/filepath"
	"strconv"
	"strings"
)

type SelectFunc func(*Sequence) float64
var SelectFuncs = map[string]SelectFunc{
	"displacement": func(seq *Sequence) float64 {
		start := seq.Items[0].Detection.Polygon().Bounds().Center()
		end := seq.Items[len(seq.Items)-1].Detection.Polygon().Bounds().Center()
		return start.Distance(end)
	},
	"length": func(seq *Sequence) float64 {
		return float64(len(seq.Items))
	},
	"duration": func(seq *Sequence) float64 {
		start := seq.Items[0].Frame
		end := seq.Items[len(seq.Items)-1].Frame
		// 5 fps
		return float64(end-start)/5
	},
}

func SelectOp(args []OpArgument, outDir string) error {
	// Parse arguments.
	predicate := args[1].String
	parts := strings.Fields(predicate)
	if len(parts) != 3 {
		return fmt.Errorf("expected select predicate to have 3 parts")
	}
	selectFunc := SelectFuncs[parts[0]]
	if selectFunc == nil {
		return fmt.Errorf("no such selection func %s", parts[0])
	}
	val, err := strconv.ParseFloat(parts[2], 64)
	if err != nil {
		return fmt.Errorf("error parsing predicate value %s: %v", parts[2], err)
	}
	evaluate := func(seq *Sequence) bool {
		v1 := selectFunc(seq)
		if parts[1] == "<" && v1 < val {
			return true
		} else if parts[1] == ">" && v1 > val {
			return true
		}
		return false
	}

	// Load the input sequences.
	var sequences []*Sequence
	inputPath := filepath.Join(args[0].DirName, "sequences.json")
	bytes, err := ioutil.ReadFile(inputPath)
	if err != nil {
		return fmt.Errorf("error loading sequences from %s: %v", inputPath, err)
	}
	if err := json.Unmarshal(bytes, &sequences); err != nil {
		return err
	}

	// Apply predicate.
	var outputs []*Sequence
	for _, seq := range sequences {
		if !evaluate(seq) {
			continue
		}
		outputs  = append(outputs, seq)
	}

	bytes, err = json.Marshal(outputs)
	if err != nil {
		return err
	}
	if err := ioutil.WriteFile(filepath.Join(outDir, "sequences.json"), bytes, 0644); err != nil {
		return err
	}
	return nil
}

func init() {
	Ops["Select"] = SelectOp
}

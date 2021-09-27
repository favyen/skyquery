package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"path/filepath"
	"sort"
)

/*
Operator takes three arguments:
- Sequences
- Image
- Mode: either "all" or "any"
Returns:
- Filtered sequences where either all or any of the detections in the sequence intersect a cell in the image that has value > 0.
*/

func IntersectOp(args []OpArgument, outDir string) error {
	// Load the input sequences.
	var sequences []*Sequence
	inputPath := filepath.Join(args[0].DirName, "sequences.json")
	bytes, err := ioutil.ReadFile(inputPath)
	if err != nil {
		return fmt.Errorf("error loading sequences from %s: %v", inputPath, err)
	}
	if err := json.Unmarshal(bytes, &sequences); err != nil {
		return fmt.Errorf("error decoding sequences: %v", err)
	}

	// Load the input matrix.
	var matrix Matrix
	inputPath = filepath.Join(args[1].DirName, "matrix.json")
	bytes, err = ioutil.ReadFile(inputPath)
	if err != nil {
		return fmt.Errorf("error loading matrix from %s: %v", inputPath, err)
	}
	if err := json.Unmarshal(bytes, &matrix); err != nil {
		return err
	}
	gridSize := matrix.GridSize

	// Set mode.
	mode := args[2].String

	// Order input sequences by the time of their last detection.
	getSequenceTime := func(seq *Sequence) int {
		return seq.Items[len(seq.Items)-1].Frame
	}
	sort.Slice(sequences, func(i, j int) bool {
		return getSequenceTime(sequences[i]) < getSequenceTime(sequences[j])
	})
	lastFrame := getSequenceTime(sequences[len(sequences)-1])

	var outputSequences []*Sequence
	var inputMatrixCounter int = 0
	var inputSequenceCounter int = 0
	curInputMatrix := make(map[[2]int]int)

	for frameIdx := 0; frameIdx < lastFrame; frameIdx++ {
		// Update input matrix state.
		for ; inputMatrixCounter < len(matrix.Observations) && matrix.Observations[inputMatrixCounter].Frame <= frameIdx; inputMatrixCounter++ {
			obs := matrix.Observations[inputMatrixCounter]
			curInputMatrix[obs.Cell] = obs.Value
		}

		// Evaluate new sequences.
		for ; inputSequenceCounter < len(sequences) && getSequenceTime(sequences[inputSequenceCounter]) <= frameIdx; inputSequenceCounter++ {
			seq := sequences[inputSequenceCounter]

			// Determine intersection success depending on mode.
			var okay bool
			if mode == "all" {
				okay = true
				for _, item := range seq.Items {
					cell := ToCell(item.Detection.Polygon().Bounds().Center(), float64(gridSize))
					if curInputMatrix[cell] <= 0 {
						okay = false
						break
					}
				}
			} else if mode == "any" {
				okay = false
				for _, item := range seq.Items {
					cell := ToCell(item.Detection.Polygon().Bounds().Center(), float64(gridSize))
					if curInputMatrix[cell] > 0 {
						okay = true
						break
					}
				}
			}

			if !okay {
				continue
			}

			outputSequences = append(outputSequences, seq)
		}
	}

	bytes, err = json.Marshal(outputSequences)
	if err != nil {
		return err
	}
	if err := ioutil.WriteFile(filepath.Join(outDir, "sequences.json"), bytes, 0644); err != nil {
		return err
	}
	return nil
}

func init() {
	Ops["Priorities"] = PrioritiesOp
}

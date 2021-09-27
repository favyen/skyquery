package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"path/filepath"
)

// The priorities operator inputs a priority rates matrix that specifies how
// quickly the "priority" at a cell in the matrix should increase. For example,
// this rate could be the variance in a forecast distribution of the current
// value at each cell. This operator computes priorities from that rate, i.e.,
// it increments the priority at each cell by the rate specified in the input
// matrix, but resets the priority to zero if the cell is visible in the frame.

func PrioritiesOp(args []OpArgument, outDir string) error {
	// Load the input matrix.
	var ratesMatrix Matrix
	inputPath := filepath.Join(args[0].DirName, "matrix.json")
	bytes, err := ioutil.ReadFile(inputPath)
	if err != nil {
		return fmt.Errorf("error loading matrix from %s: %v", inputPath, err)
	}
	if err := json.Unmarshal(bytes, &ratesMatrix); err != nil {
		return err
	}
	gridSize := ratesMatrix.GridSize

	// Load frame bounds.
	var frames []Frame
	bytes, err = ioutil.ReadFile(filepath.Join(Config.DataDir, "align-out.json"))
	if err != nil {
		return fmt.Errorf("error loading frame bounds: %v", err)
	}
	if err := json.Unmarshal(bytes, &frames); err != nil {
		return fmt.Errorf("error decoding frame bounds: %v", err)
	}

	matrixObservations := []MatrixObservation{}
	curObservations := make(map[[2]int]*MatrixObservation)
	inputObsCounter := 0

	for frameIdx, frame := range frames {
		for cell := range GetCellsInFrame(frame, float64(gridSize)) {
			if curObservations[cell] == nil || curObservations[cell].Value == 0 {
				continue
			}
			obs := MatrixObservation{
				Cell: cell,
				Frame: frameIdx,
				Value: 0,
				Metadata: "",
			}
			curObservations[cell] = &obs
			matrixObservations = append(matrixObservations, obs)
		}

		for ; inputObsCounter < len(ratesMatrix.Observations) && ratesMatrix.Observations[inputObsCounter].Frame == frameIdx; inputObsCounter++ {
			ratesObs := ratesMatrix.Observations[inputObsCounter]

			// Increment priority by ratesObs.Value, but set to zero if the frame is visible.
			prevObs := curObservations[ratesObs.Cell]
			var priority int = 0
			if prevObs != nil {
				priority = prevObs.Value
			}
			priority += ratesObs.Value
			if IsCellInFrame(ratesObs.Cell, frame, float64(gridSize)) {
				priority = 0
			}
			obs := MatrixObservation{
				Cell: ratesObs.Cell,
				Frame: frameIdx,
				Value: priority,
				Metadata: "",
			}
			curObservations[obs.Cell] = &obs
			matrixObservations = append(matrixObservations, obs)
		}
	}

	matrix := Matrix{
		GridSize: gridSize,
		Observations: matrixObservations,
	}
	bytes, err = json.Marshal(matrix)
	if err != nil {
		return err
	}
	if err := ioutil.WriteFile(filepath.Join(outDir, "matrix.json"), bytes, 0644); err != nil {
		return err
	}
	return nil
}

func init() {
	Ops["Priorities"] = PrioritiesOp
}

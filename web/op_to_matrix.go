package main

import (
	"github.com/mitroadmaps/gomapinfer/common"

	"encoding/json"
	"fmt"
	"io/ioutil"
	"math"
	"path/filepath"
	"strconv"
	"strings"
)

type Frame [][2]float64

func (f Frame) Polygon() common.Polygon {
	poly := common.Polygon{}
	for _, p := range f {
		poly = append(poly, common.Point{float64(p[0]), float64(p[1])})
	}
	return poly
}

type MatrixObservation struct {
	Cell [2]int
	Frame int
	Value int
	Metadata string
}

type Matrix struct {
	GridSize int
	Observations []MatrixObservation
}

/*
ToMatrix strategy:
- This operator takes an aggregation function of the form:
	func(cell, frame, sequences)
  The function should return a value given the cell, frame, and sequences intersecting the cell.
- ToMatrix chooses which frame to use intelligently:
  * For each cell, we maintain a struct cellStatus{bestFrame, sequences, distance}.
  * We choose bestFrame based on maximizing the minimum distance from cell boundaries to the frame boundaries.
  * cellStatus.sequences contains the sequences that we got at that frame.
  * Once cell is no longer visible in the video, we run the aggregation function on bestFrame and create a new matrix entry.
  * The timestamp of the entry is the time when the cell leaves the field of view.
*/

type ToMatrixAggFunc func(cell [2]int, prev int, metadata string, frame Frame, seqs []*Sequence) (int, string)
var ToMatrixAggFuncs = map[string]ToMatrixAggFunc{
	"count": func(cell [2]int, prev int, metadata string, frame Frame, seqs []*Sequence) (int, string) {
		return len(seqs), ""
	},
	"count_sum": func(cell [2]int, prev int, metadata string, frame Frame, seqs []*Sequence) (int, string) {
		return prev + len(seqs), ""
	},
	"count_old_sum": func(cell [2]int, prev int, metadata string, frame Frame, seqs []*Sequence) (int, string) {
		var prevIDs []int
		JsonUnmarshal([]byte(metadata), &prevIDs)
		prevIDSet := make(map[int]bool)
		for _, id := range prevIDs {
			prevIDSet[id] = true
		}
		var curIDs []int
		curIDSet := make(map[int]bool)
		var countNew int
		for _, seq := range seqs {
			curIDs = append(curIDs, seq.ID)
			curIDSet[seq.ID] = true
			if !prevIDSet[seq.ID] {
				countNew++
			}
		}
		var countOld int
		for _, id := range prevIDs {
			if !curIDSet[id] {
				countOld++
			}
		}
		return prev + countOld, string(JsonMarshal(curIDs))
	},
	"avg_speed": func(cell [2]int, prev int, metadata string, frame Frame, seqs []*Sequence) (int, string) {
		metaParts := strings.Split(metadata, ",")
		var sum, count float64
		if len(metaParts) == 2 {
			sum, _ = strconv.ParseFloat(metaParts[0], 64)
			count, _ = strconv.ParseFloat(metaParts[1], 64)
		}
		for _, seq := range seqs {
			first := seq.Items[0]
			last := seq.Items[len(seq.Items)-1]
			if last.Frame-first.Frame <= 0 {
				continue
			}
			d := first.Detection.Polygon().Bounds().Center().Distance(last.Detection.Polygon().Bounds().Center())
			t := float64(last.Frame-first.Frame)
			speed := d / t
			sum += speed
			count++
		}
		if count == 0 {
			return 0, ""
		} else {
			return int(sum / count), fmt.Sprintf("%v,%v", sum, count)
		}
	},
}

func ToCell(p common.Point, gridSize float64) [2]int {
	return [2]int{
		int(math.Floor(p.X / gridSize)),
		int(math.Floor(p.Y / gridSize)),
	}
}

func GetCellRect(cell [2]int, gridSize float64) common.Rectangle {
	cellPoint := common.Point{float64(cell[0]), float64(cell[1])}
	return common.Rectangle{
		cellPoint.Scale(gridSize),
		cellPoint.Add(common.Point{1, 1}).Scale(gridSize),
	}
}

func IsCellInFrame(cell [2]int, frame Frame, gridSize float64) bool {
	cellRect := GetCellRect(cell, gridSize)
	for _, p := range cellRect.ToPolygon() {
		if !frame.Polygon().Contains(p) {
			return false
		}
	}
	return true
}

// Returns map from cells visible in current frame to the distances from
// those cells to the frame boundaries
func GetCellsInFrame(frame Frame, gridSize float64) map[[2]int]float64 {
	frameRect := frame.Polygon().Bounds()
	startCell := ToCell(frameRect.Min, gridSize)
	endCell := ToCell(frameRect.Max, gridSize)
	frameCells := make(map[[2]int]float64)
	processCell := func(cell [2]int) {
		if !IsCellInFrame(cell, frame, gridSize) {
			return
		}
		cellRect := GetCellRect(cell, gridSize)
		var worstDistance float64 = -1
		for _, cellSegment := range cellRect.ToPolygon().Segments() {
			for _, frameSegment := range frame.Polygon().Segments() {
				d := cellSegment.DistanceToSegment(frameSegment)
				if worstDistance == -1 || d < worstDistance {
					worstDistance = d
				}
			}
		}
		frameCells[cell] = worstDistance
	}
	for i := startCell[0]; i <= endCell[0]; i++ {
		for j := startCell[1]; j <= endCell[1]; j++ {
			processCell([2]int{i, j})
		}
	}
	return frameCells
}

func ToMatrixOp(args []OpArgument, outDir string) error {
	// Parse arguments.
	var operands struct {
		Func string
		GridSize int
		IgnoreZero bool
		UnionSeqs bool
	}
	err := json.Unmarshal([]byte(args[1].String), &operands)
	if err != nil {
		return fmt.Errorf("error decoding operands %s: %v", args[1].String, err)
	}
	if operands.Func == "" {
		operands.Func = "count"
	}
	if operands.GridSize == 0 {
		operands.GridSize = 32
	}
	aggFunc := ToMatrixAggFuncs[operands.Func]

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

	// Load frame bounds.
	var frames []Frame
	bytes, err = ioutil.ReadFile(filepath.Join(Config.DataDir, "align-out.json"))
	if err != nil {
		return fmt.Errorf("error loading frame bounds: %v", err)
	}
	if err := json.Unmarshal(bytes, &frames); err != nil {
		return fmt.Errorf("error decoding frame bounds: %v", err)
	}

	// Status is used to select the best frame for each cell, where the cell
	// is closest to the center.
	type cellStatus struct {
		bestFrame Frame
		sequences map[int]*Sequence
		distance float64
	}
	cellStatuses := make(map[[2]int]*cellStatus)

	// Gets sequences that are inside a given cell.
	// seqLocations contains the point of each sequence at the current timestep.
	getRelevantSequences := func(seqs []*Sequence, cell [2]int, seqLocations map[int]*common.Point) map[int]*Sequence {
		cellRect := GetCellRect(cell, float64(operands.GridSize))
		relevantSeqs := make(map[int]*Sequence)
		for _, seq := range seqs {
			location := seqLocations[seq.ID]
			if location == nil {
				continue
			}
			if !cellRect.Contains(*location) {
				continue
			}
			relevantSeqs[seq.ID] = seq
		}
		return relevantSeqs
	}

	frameSequences := make([][]*Sequence, len(frames))
	for _, seq := range sequences {
		for frameIdx := seq.Items[0].Frame; frameIdx <= seq.Items[len(seq.Items)-1].Frame; frameIdx++ {
			frameSequences[frameIdx] = append(frameSequences[frameIdx], seq)
		}
	}
	matrixObservations := []MatrixObservation{}
	curObservations := make(map[[2]int]*MatrixObservation)

	// Build matrix.
	for frameIdx, frame := range frames {
		seqs := frameSequences[frameIdx]
		frameCells := GetCellsInFrame(frame, float64(operands.GridSize))

		// get location of sequences at this frame
		seqLocations := make(map[int]*common.Point)
		for _, seq := range seqs {
			seqLocations[seq.ID] = seq.LocationAt(frameIdx)
		}

		fmt.Printf("[to_matrix] got %d cells in frame with %d seq locations, %d seqs\n", len(frameCells), len(seqLocations), len(seqs))

		// update cell status based on frameCells
		for cell, distance := range frameCells {
			status := cellStatuses[cell]
			// If existing cellStatus has higher distance to frame bounds, then retain it.
			if status != nil && status.distance > distance {
				if operands.UnionSeqs {
					relevantSeqs := getRelevantSequences(seqs, cell, seqLocations)
					for _, seq := range relevantSeqs {
						cellStatuses[cell].sequences[seq.ID] = seq
					}
				}
				continue
			}
			relevantSeqs := getRelevantSequences(seqs, cell, seqLocations)
			if operands.IgnoreZero && len(relevantSeqs) == 0 {
				continue
			}
			if operands.UnionSeqs && cellStatuses[cell] != nil {
				for _, seq := range cellStatuses[cell].sequences {
					relevantSeqs[seq.ID] = seq
				}
			}
			cellStatuses[cell] = &cellStatus{
				bestFrame: frame,
				sequences: relevantSeqs,
				distance: distance,
			}
		}

		// run aggregation function on best frames of cells that left
		for cell, status := range cellStatuses {
			if _, ok := frameCells[cell]; ok {
				continue
			}
			fmt.Printf("[to_matrix] frame %d: adding observation at cell %v\n", frameIdx, cell)
			prevObs := curObservations[cell]
			var prev int = 0
			var metadata string
			if prevObs != nil {
				prev = prevObs.Value
				metadata = prevObs.Metadata
			}
			var sequences []*Sequence
			for _, seq := range status.sequences {
				sequences = append(sequences, seq)
			}
			val, metadata := aggFunc(cell, prev, metadata, status.bestFrame, sequences)
			obs := MatrixObservation{
				Cell: cell,
				Frame: frameIdx,
				Value: val,
				Metadata: metadata,
			}
			curObservations[cell] = &obs
			matrixObservations = append(matrixObservations, obs)
			delete(cellStatuses, cell)
		}
	}

	matrix := Matrix{
		GridSize: operands.GridSize,
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
	Ops["ToMatrix"] = ToMatrixOp
}

package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math"
	"path/filepath"
)

// Simple operator for forecasting the value and approximation of a "variance"
// at each cell. This assumes a cyclic pattern, e.g. where the user expects
// consistent daily changes in the input matrix value.

// Most recent processed sample recorded at a cell.
type PrevSample struct {
	interval int
	val int
}

type Prediction struct {
	Val float64
	Stddev float64
}

func getMean(a []float64) float64 {
	if len(a) == 0 {
		return 0
	}
	var sum float64 = 0
	for _, x := range a {
		sum += x
	}
	return sum / float64(len(a))
}

func getStddev(a []float64, max int) float64 {
	if len(a) == 0 {
		return float64(max)
	}
	mean := getMean(a)
	var sqdevsum float64 = 0
	for _, x := range a {
		d := x - mean
		sqdevsum += d * d
	}
	stddev := math.Sqrt(sqdevsum / float64(len(a)))
	return stddev
}

func ForecastOp(args []OpArgument, outDir string) error {
	// Parse arguments.
	var operands struct {
		// Run the forecasting model every Frequency frames.
		Frequency int

		// The number of intervals over which changes are expected to be cyclic.
		// One interval is equal to Frequency frames.
		// For example, for daily changes, and with 5 fps video, we might set
		// Frequency=15*60*5=4500 to forecast values every 15 minutes.
		// Then, setting Period=24*4=96 would specify that the period of the
		// expected cyclic patterns is daily (96 15-min intervals per day).
		Period int
	}
	err := json.Unmarshal([]byte(args[1].String), &operands)
	if err != nil {
		return fmt.Errorf("error decoding operands %s: %v", args[1].String, err)
	}

	// Load the input matrix.
	var inputMatrix Matrix
	inputPath := filepath.Join(args[0].DirName, "matrix.json")
	bytes, err := ioutil.ReadFile(inputPath)
	if err != nil {
		return fmt.Errorf("error loading matrix from %s: %v", inputPath, err)
	}
	if err := json.Unmarshal(bytes, &inputMatrix); err != nil {
		return err
	}
	gridSize := inputMatrix.GridSize

	// Get unique cells.
	cells := make(map[[2]int]bool)
	for _, obs := range inputMatrix.Observations {
		cells[obs.Cell] = true
	}

	// Map: cell -> (cycle, histsize) -> samples.
	// Cell: matrix cell that the samples correspond to.
	// Cycle: the cycle between 0 and Period for this sample.
	// HistSize: history size.
	// The samples describe the amount of change in the value between Cycle-HistSize and Cycle.
	cyclicSamples := make(map[[2]int]map[[2]int][]float64)

	// Previous sample value and the interval when that value was captured, at each cell.
	prevSamples := make(map[[2]int][]PrevSample)

	// Output observations.
	matrixObservations := []MatrixObservation{}

	stddevs := make(map[[2]int]float64)
	var max int

	endIdx := inputMatrix.Observations[len(inputMatrix.Observations)-1].Frame
	inputObsCounter := 0

	for interval := 0; interval <= endIdx/operands.Frequency; interval++ {
		cycle := interval % operands.Period
		nextIntervalFrame := (interval+1) * operands.Frequency

		// Process input observations from the current interval.
		for ; inputObsCounter < len(inputMatrix.Observations) && inputMatrix.Observations[inputObsCounter].Frame < nextIntervalFrame; inputObsCounter++ {
			obs := inputMatrix.Observations[inputObsCounter]
			cell := obs.Cell
			value := obs.Value

			if cyclicSamples[cell] == nil {
				cyclicSamples[cell] = make(map[[2]int][]float64)
			}
			if len(prevSamples[cell]) > 0 {
				// Tabulate previous values for the last (period) intervals.
				// We use linear interpolation to get values in between samples when needed.
				var prevTable []float64
				curSample := PrevSample{
					interval: interval,
					val: value,
				}
				prevIdx := len(prevSamples[cell]) - 1
				for histsize := 0; histsize < operands.Period; histsize++ {
					// Get value at (interval - histsize).
					prevSample := prevSamples[cell][prevIdx]
					wantInterval := interval - histsize
					curWeight := wantInterval - prevSample.interval
					prevWeight := curSample.interval - wantInterval
					interp := float64(curWeight * curSample.val + prevWeight * prevSample.val) / float64(curWeight + prevWeight)
					prevTable = append(prevTable, interp)

					// If we got sample(s) at (interval - histsize), update curSample.
					// This way, for the next histsize, we will interpolate between a more accurate sample.
					for prevIdx >= 0 && wantInterval == prevSamples[cell][prevIdx].interval {
						curSample = prevSamples[cell][prevIdx]
						prevIdx--
					}
					if prevIdx < 0 {
						break
					}
				}

				// Extend the table so we don't need to worry about running past the end.
				for len(prevTable) < 2*operands.Period {
					prevTable = append(prevTable, prevTable[len(prevTable) - 1])
				}

				// Update cyclicSamples to reflect the new information.
				// For each interval T between that of the previous sample and now, we update
				// cyclicSamples with the estimated amount of change between T-histsize and T,
				// for each histsize that's possible to compute given prevTable (i.e.), at
				// most up to 2*operands.Period.
				prevSample := prevSamples[cell][len(prevSamples[cell]) - 1]
				for curInterval := prevSample.interval + 1; curInterval <= interval; curInterval++ {
					age := interval - curInterval
					for histsize := 1; histsize < len(prevTable) - age; histsize++ {
						curCycle := curInterval % operands.Period
						cur := prevTable[age]
						prev := prevTable[age + histsize]
						rate := cur - prev
						k := [2]int{curCycle, histsize}
						cyclicSamples[cell][k] = append(cyclicSamples[cell][k], rate)
					}
				}
			}

			prevSamples[cell] = append(prevSamples[cell], PrevSample{
				interval: interval,
				val: value,
			})

			if value > max {
				max = value
			}
		}

		// Add predictions for all cells to the output matrix.
		// We put the prediction in metadata and the stddev in the matrix value.
		// (Really we should output two matrices, but that isn't supported in this version of the code.)
		for cell := range cells {
			prevSample := prevSamples[cell][len(prevSamples[cell]) - 1]

			value, stddev := func() (float64, float64) {
				// If we just get an observation at this cell on the current interval,
				// then we should not observe it again immediately -- so set "variance"
				// to zero.
				if prevSample.interval == interval {
					stddevs[cell] = 0
					return float64(prevSample.val), 0
				}

				// Use the samples of rate changes between prevSample and now to determine
				// the variance at this cell.
				histsize := interval - prevSample.interval
				if histsize >= operands.Period {
					histsize = operands.Period - 1
				}
				k := [2]int{cycle, histsize}

				// Get stddev.
				if len(cyclicSamples[cell][k]) < 3 {
					// If there were fewer than three samples so far, then add a large amount to stddevs.
					// This means we really want to prioritize more observations at this cell before
					// trusting our variance estimates.
					stddevs[cell] += 99
				} else {
					stddevs[cell] = getStddev(cyclicSamples[cell][k], max)
				}

				// Get mean.
				var value float64
				if len(cyclicSamples[cell][k]) < 1 {
					value = float64(prevSample.val)
				} else {
					change := getMean(cyclicSamples[cell][k])
					value = float64(prevSample.val) + change
					if value < 0 {
						value = 0
					}
				}

				return value, stddevs[cell]
			}()

			prediction := Prediction{
				Val: value,
				Stddev: stddev,
			}
			bytes, err := json.Marshal(prediction)
			if err != nil {
				panic(err)
			}
			matrixObservations = append(matrixObservations, MatrixObservation{
				Cell: cell,
				Frame: interval * operands.Frequency,
				Value: int(value),
				Metadata: string(bytes),
			})
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
	Ops["Forecast"] = ForecastOp
}

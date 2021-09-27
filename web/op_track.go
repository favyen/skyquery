package main

import (
	"github.com/mitroadmaps/gomapinfer/common"
	goslgraph "./munkres"

	"encoding/json"
	"fmt"
	"io/ioutil"
	"path/filepath"
)

type SequenceItem struct {
	Detection Detection
	Frame int
}

type Sequence struct {
	ID int
	Items []SequenceItem
}

// Returns location of this sequence at specified time,
// or nil if the time is before first member or after last member.
// To compute location, we take center-point of rectangle bound of
// detections before/after the time, and average those points based
// on the time difference.
func (seq Sequence) LocationAt(t int) *common.Point {
	if t < seq.Items[0].Frame || t > seq.Items[len(seq.Items)-1].Frame {
		return nil
	} else if len(seq.Items) == 1 {
		p := seq.Items[0].Detection.Polygon().Bounds().Center()
		return &p
	}
	var item1, item2 SequenceItem
	for i := 0; i < len(seq.Items) - 1; i++ {
		if seq.Items[i+1].Frame < t {
			continue
		}
		item1 = seq.Items[i]
		item2 = seq.Items[i+1]
		break
	}
	p1 := item1.Detection.Polygon().Bounds().Center()
	p2 := item2.Detection.Polygon().Bounds().Center()
	t1 := t - item1.Frame
	t2 := item2.Frame - t
	if t1 == 0 {
		return &p1
	} else if t2 == 0 {
		return &p2
	}
	v := p2.Sub(p1)
	location := p1.Add(v.Scale(float64(t1) / float64(t1 + t2)))
	return &location
}

func TrackOp(args []OpArgument, outDir string) error {
	// Load the input detections.
	var detections [][]Detection
	detectionPath := filepath.Join(args[0].DirName, "detect.json")
	bytes, err := ioutil.ReadFile(detectionPath)
	if err != nil {
		return fmt.Errorf("error loading detections from %s: %v", detectionPath, err)
	}
	if err := json.Unmarshal(bytes, &detections); err != nil {
		return err
	}

	sequences := []*Sequence{}
	activeSequences := make(map[int]*Sequence)

	for frameIdx, dlist := range detections {
		detectionMap := make(map[int]*Detection)
		for i, detection := range dlist {
			x := detection
			detectionMap[i] = &x
		}
		matches := hungarianMatcher(activeSequences, detectionMap)
		for seqID, detection := range matches {
			sequences[seqID].Items = append(sequences[seqID].Items, SequenceItem{
				Detection: *detection,
				Frame: frameIdx,
			})
		}

		// new sequences for unmatched detections
		for _, detection := range detectionMap {
			seq := &Sequence{
				ID: len(sequences),
				Items: []SequenceItem{{
					Detection: *detection,
					Frame: frameIdx,
				}},
			}
			activeSequences[seq.ID] = seq
			sequences = append(sequences, seq)
		}

		// remove old active sequences
		for id, seq := range activeSequences {
			lastTime := seq.Items[len(seq.Items)-1].Frame
			if frameIdx - lastTime < 10 {
				continue
			}
			delete(activeSequences, id)
		}
	}

	bytes, err = json.Marshal(sequences)
	if err != nil {
		return err
	}
	if err := ioutil.WriteFile(filepath.Join(outDir, "sequences.json"), bytes, 0644); err != nil {
		return err
	}
	return nil
}

// Returns map from tracks to detection that should be added corresponding to that track.
// Also removes detections from the map that matched with a track.
func hungarianMatcher(sequences map[int]*Sequence, detections map[int]*Detection) map[int]*Detection {
	if len(sequences) == 0 || len(detections) == 0 {
		return nil
	}

	var sequenceList []*Sequence
	var sequenceIDs []int
	for id, seq := range sequences {
		sequenceList = append(sequenceList, seq)
		sequenceIDs = append(sequenceIDs, id)
	}
	var detectionList []*Detection
	var detectionIDs []int
	for id, detection := range detections {
		detectionList = append(detectionList, detection)
		detectionIDs = append(detectionIDs, id)
	}

	// create cost matrix for hungarian algorithm
	// rows: existing sequences (sequenceList)
	// cols: current detections (detectionList)
	// values: 1-IoU if overlap is non-zero, or 10 otherwise
	costMatrix := make([][]float64, len(sequenceList))
	for i, seq := range sequenceList {
		costMatrix[i] = make([]float64, len(detectionList))
		seqRect := seq.Items[len(seq.Items) - 1].Detection.Polygon().Bounds()

		for j, detection := range detectionList {
			curRect := detection.Polygon().Bounds()
			iou := seqRect.IOU(curRect)
			var cost float64
			if iou > 0.99 {
				cost = 0.01
			} else if iou > 0.1 {
				cost = 1 - iou
			} else {
				cost = 10
			}
			costMatrix[i][j] = cost
		}
	}

	munkres := &goslgraph.Munkres{}
	munkres.Init(len(sequenceList), len(detectionList))
	munkres.SetCostMatrix(costMatrix)
	munkres.Run()

	matches := make(map[int]*Detection)
	for i, j := range munkres.Links {
		if j < 0 || costMatrix[i][j] > 0.9 {
			continue
		}
		matches[sequenceIDs[i]] = detectionList[j]
		delete(detections, detectionIDs[j])
	}
	return matches
}

func init() {
	Ops["Track"] = TrackOp
}

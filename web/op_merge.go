package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
)

// Minimum number of consecutive frames where sequence end is visible in the field of view
// to qualify for a gap (sequence termination).
const SeqMergeGapThreshold int = 10

// Minimum distance from edge of frame for counting gaps.
const SeqMergeGapPadding float64 = 50

func MergeOp(args []OpArgument, outDir string) error {
	var operands struct {
		Mode string

		// Maximum distance of next seq start poly from previous seq end poly.
		// 40 for parked cars
		// 150 for hazards
		DistanceThreshold float64
	}
	err := json.Unmarshal([]byte(args[1].String), &operands)
	if err != nil {
		return fmt.Errorf("error decoding operands %s: %v", args[1].String, err)
	}

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

	cachedImageSimilarities := make(map[[2]int]float64)

	// map from parent sequence ID -> our merged sequence
	parentSeqMap := make(map[int]*Sequence)

	// Number of frames that the last point of a sequence was contained in the field of view.
	// Resets to 0 if goes out of the view.
	// We use this as follows: If an active sequence was visible in the field of view for at
	// least SeqMergeGapThreshold frames, but wasn't seen in the input sequence table, then
	// we terminate the sequence (mark inactive).
	seqVisibleFrames := make(map[int]int)

	activeSequences := make(map[int]*Sequence)
	outputs := []*Sequence{}

	// return sequences that end before the specified frame
	// these are candidates for termination
	getCandidateSequences := func(frameIdx int) map[int]*Sequence {
		seqs := make(map[int]*Sequence)
		for _, seq := range activeSequences {
			endTime := seq.Items[len(seq.Items)-1].Frame
			if endTime >= frameIdx {
				continue
			}
			seqs[seq.ID] = seq
		}
		return seqs
	}

	getDetectionDistanceToFrame := func(frame Frame, detection Detection) float64 {
		p := detection.Polygon().Bounds().Center()
		if !frame.Polygon().Contains(p) {
			return -1
		}
		var d float64 = -1
		for _, segment := range frame.Polygon().Segments() {
			segD := segment.Distance(p)
			if d == -1 || segD < d {
				d = segD
			}
		}
		return d
	}

	// Returns active sequences where the previous input sequence ended within the provided frame bounds.
	getSequencesEndingInFrame := func(frame Frame) map[int]*Sequence {
		matchSeqs := make(map[int]*Sequence)
		for _, seq := range activeSequences {
			detection := seq.Items[len(seq.Items)-1].Detection
			d := getDetectionDistanceToFrame(frame, detection)
			if d == -1 || d < SeqMergeGapPadding {
				continue
			}
			matchSeqs[seq.ID] = seq
		}
		return matchSeqs
	}

	// update seqVisibleFrames for this frame
	// returns list of sequences that got a gap
	updateSeqVisibleFrames := func(frame Frame, frameIdx int) []*Sequence {
		matchSeqs := getSequencesEndingInFrame(frame)

		// if seq in matchSeqs, increment frame counter
		// else, reset status
		var gapSeqs []*Sequence
		for _, seq := range getCandidateSequences(frameIdx) {
			if matchSeqs[seq.ID] != nil {
				seqVisibleFrames[seq.ID]++
			} else {
				if seqVisibleFrames[seq.ID] >= SeqMergeGapThreshold {
					gapSeqs = append(gapSeqs, seq)
				}
				delete(seqVisibleFrames, seq.ID)
			}
		}

		return gapSeqs
	}

	// return first or last detection at least SeqMergeGapPadding away from frame
	findPaddedDetection := func(seq *Sequence, first bool) SequenceItem {
		items := seq.Items
		if !first {
			nitems := make([]SequenceItem, len(items))
			for i := range items {
				nitems[i] = items[len(items) - i - 1]
			}
			items = nitems
		}
		for _, item := range items {
			frame := frames[item.Frame]
			d := getDetectionDistanceToFrame(frame, item.Detection)
			if d >= SeqMergeGapPadding {
				return item
			}
		}
		return items[0]
	}

	frameSequences := make([][]*Sequence, len(frames))
	for _, seq := range sequences {
		frameIdx := seq.Items[0].Frame
		frameSequences[frameIdx] = append(frameSequences[frameIdx], seq)
	}

	for frameIdx, frame := range frames {
		// merge seqs into candidates
		for _, parentSeq := range frameSequences[frameIdx] {
			if parentSeqMap[parentSeq.ID] != nil {
				continue
			}
			parentBegins := parentSeq.Items[0]
			if len(parentSeq.Items) >= 4 {
				parentBegins = parentSeq.Items[3]
			}
			parentPoint := parentBegins.Detection.Polygon().Bounds().Center()

			var bestMergeSequence *Sequence
			var bestDistance float64

			for _, mySeq := range activeSequences {
				myEnds := mySeq.Items[len(mySeq.Items)-1]

				// only for parked cars!
				if len(mySeq.Items) >= 4 {
					myEnds = mySeq.Items[len(mySeq.Items)-4]
				}

				myPoint := myEnds.Detection.Polygon().Bounds().Center()
				d := parentPoint.Distance(myPoint)
				if d > operands.DistanceThreshold {
					continue
				} else if parentBegins.Frame < myEnds.Frame {
					continue
				}

				if operands.Mode == "image_similarity" {
					// use external python script to verify that image similarity is close
					// first get last/first detections that are SeqMergeGapPadding away from their frames
					var similarity float64
					k := [2]int{parentSeq.ID, mySeq.ID}
					if _, ok := cachedImageSimilarities[k]; ok {
						similarity = cachedImageSimilarities[k]
					} else {
						item1 := findPaddedDetection(parentSeq, true)
						item2 := findPaddedDetection(mySeq, false)
						similarity = getImageSimilarity(item1.Frame, item1.Detection, item2.Frame, item2.Detection)
						cachedImageSimilarities[k] = similarity
						log.Printf("[merge] %d/%d %v %v\n", frameIdx, len(frames), k, similarity)
					}
					if similarity < 0.15 {
						continue
					}
				}

				if bestMergeSequence == nil || d < bestDistance {
					bestMergeSequence = mySeq
					bestDistance = d
				}
			}

			if bestMergeSequence != nil {
				for _, item := range parentSeq.Items {
					bestMergeSequence.Items = append(bestMergeSequence.Items, item)
				}
				//bestMergeSequence.AddMetadata(fmt.Sprintf("%d", parentSeq.ID), frame.Time)
				parentSeqMap[parentSeq.ID] = bestMergeSequence
				delete(seqVisibleFrames, bestMergeSequence.ID)
			}
		}

		// Create new sequences for parent sequences that were not merged.
		for _, parentSeq := range frameSequences[frameIdx] {
			if parentSeqMap[parentSeq.ID] != nil {
				continue
			}

			mySeq := &Sequence{
				ID: parentSeq.ID,
				Items: append([]SequenceItem{}, parentSeq.Items...),
			}
			//mySeq.AddMetadata(fmt.Sprintf("%d", parentSeq.ID), parentSeq.Time)
			parentSeqMap[parentSeq.ID] = mySeq
			activeSequences[mySeq.ID] = mySeq
			outputs = append(outputs, mySeq)
		}

		// Terminate sequences that should have been visible in the frame but weren't
		// seen, and that are no longer in the frame now.
		gapSeqs := updateSeqVisibleFrames(frame, frameIdx)
		for _, mySeq := range gapSeqs {
			delete(activeSequences, mySeq.ID)
		}
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

func getImageSimilarity(frameIdx1 int, detection1 Detection, frameIdx2 int, detection2 Detection) float64 {
	poly1 := PointsToPolyString(detection1.OrigPoints)
	poly2 := PointsToPolyString(detection2.OrigPoints)
	cmd := exec.Command(Config.Python, "web/seq-merge-imagediff.py", Config.VideoDir, strconv.Itoa(frameIdx1), strconv.Itoa(frameIdx2), poly1, poly2)
	bytes, err := cmd.CombinedOutput()
	if err != nil {
		fmt.Println(string(bytes))
		fmt.Println("warning!! image similarity error")
		//panic(err)
		return 0
	}
	output := strings.TrimSpace(string(bytes))
	lines := strings.Split(output, "\n")
	lastLine := lines[len(lines)-1]
	if strings.Contains(lastLine, "bad") {
		return 0
	}
	similarity, err := strconv.ParseFloat(lastLine, 64)
	if err != nil {
		fmt.Println(output)
		panic(err)
	}
	return similarity
}

func init() {
	Ops["Merge"] = MergeOp
}

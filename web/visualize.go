package main

import (
	"github.com/mitroadmaps/gomapinfer/common"
	"github.com/mitroadmaps/gomapinfer/image"

	"encoding/json"
	"io/ioutil"
	"log"
	"path/filepath"
)

// Visualize the outputs of an operation that are stored in the given directory.
func Visualize(dir string) error {
	log.Printf("[visualize] loading ortho-image")
	ortho := image.ReadImage(filepath.Join(Config.DataDir, "ortho.jpg"))

	files, err := ioutil.ReadDir(dir)
	if err != nil {
		return err
	}
	for _, fi := range files {
		if fi.Name() == "detect.json" {
			log.Printf("[visualize] drawing detections")
			var detections [][]Detection
			bytes, err := ioutil.ReadFile(filepath.Join(dir, fi.Name()))
			if err != nil {
				return err
			}
			if err := json.Unmarshal(bytes, &detections); err != nil {
				return err
			}

			for _, dlist := range detections {
				for _, d := range dlist {
					xsum, ysum := 0, 0
					for _, point := range d.Points {
						xsum += point[0]
						ysum += point[1]
					}
					cx, cy := xsum/len(d.Points), ysum/len(d.Points)
					image.DrawRect(ortho, cx, cy, 1, [3]uint8{255, 255, 0})
				}
			}
		} else if fi.Name() == "sequences.json" {
			log.Printf("[visualize] drawing sequences")
			var sequences []*Sequence
			bytes, err := ioutil.ReadFile(filepath.Join(dir, fi.Name()))
			if err != nil {
				return err
			}
			if err := json.Unmarshal(bytes, &sequences); err != nil {
				return err
			}

			for _, seq := range sequences {
				prevCenter := seq.Items[0].Detection.Polygon().Bounds().Center()
				for _, item := range seq.Items[1:] {
					curCenter := item.Detection.Polygon().Bounds().Center()
					for _, p := range common.DrawLineOnCells(int(prevCenter.X), int(prevCenter.Y), int(curCenter.X), int(curCenter.Y), len(ortho), len(ortho[0])) {
						image.DrawRect(ortho, p[0], p[1], 0, [3]uint8{255, 255, 0})
					}
					prevCenter = curCenter
				}
			}
		} else if fi.Name() == "matrix.json" {
			log.Printf("[visualize] drawing matrix")
			var matrix Matrix
			bytes, err := ioutil.ReadFile(filepath.Join(dir, fi.Name()))
			if err != nil {
				return err
			}
			if err := json.Unmarshal(bytes, &matrix); err != nil {
				return err
			}

			cells := make(map[[2]int]int)
			for i := 0; i <= len(ortho)/matrix.GridSize; i++ {
				for j := 0; j <= len(ortho[0])/matrix.GridSize; j++ {
					cells[[2]int{i, j}] = 0
				}
			}

			// Populate cells with observations in the matrix.
			var min, max int
			for _, obs := range matrix.Observations {
				cells[obs.Cell] = obs.Value
				min = obs.Value
				max = obs.Value
			}
			for _, val := range cells {
				if val < min {
					min = val
				}
				if val > max {
					max = val
				}
			}
			min = 0
			max = 1

			// Returns color given value by normalized based on min/max.
			normalize := func(val int) uint8 {
				norm := float64(val - min) / float64(max - min)
				if norm < 0 {
					norm = 0
				} else if norm > 1 {
					norm = 1
				}
				return uint8(norm * 255)
			}

			// Draw rectangles.
			for cell, val := range cells {
				normVal := normalize(val)
				center := [2]int{
					cell[0]*matrix.GridSize + matrix.GridSize/2,
					cell[1]*matrix.GridSize + matrix.GridSize/2,
				}
				//image.DrawRect(ortho, center[0], center[1], matrix.GridSize/2, [3]uint8{normVal, normVal, normVal})
				//image.DrawTransparent(ortho, center[0], center[1], matrix.GridSize/2, [3]int{int(normVal), -1, -1})
				if normVal > 200 {
					image.DrawRect(ortho, center[0], center[1], matrix.GridSize/2, [3]uint8{255, 0, 0})
				}
			}
		}
	}

	log.Printf("[visualize] saving visualization")
	image.WriteImage(filepath.Join(Config.DataDir, "out.jpg"), ortho)
	return nil
}

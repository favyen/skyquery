package main

// modified from skyql-subsystem/run-yolo.go

import (
	"github.com/mitroadmaps/gomapinfer/common"

	"bufio"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

type Detection struct {
	Points [][2]int
}

func main() {
	cfgPath := os.Args[1]
	weightsPath := os.Args[2]
	videoPath := os.Args[3]
	outPath := os.Args[4]

	c := exec.Command("./darknet", "detect", cfgPath, weightsPath, "-thresh", "0.3")
	c.Dir = "darknet/"
	stdin, err := c.StdinPipe()
	if err != nil {
		panic(err)
	}
	stderr, err := c.StderrPipe()
	if err != nil {
		panic(err)
	}
	stdout, err := c.StdoutPipe()
	if err != nil {
		panic(err)
	}
	go func() {
		r := bufio.NewReader(stderr)
		for {
			line, err := r.ReadString('\n')
			if err != nil {
				panic(err)
			}
			line = strings.TrimSpace(line)
			fmt.Println("[yolo] [stderr] " + line)
		}
	}()
	r := bufio.NewReader(stdout)
	if err := c.Start(); err != nil {
		panic(err)
	}

	files, err := ioutil.ReadDir(videoPath)
	if err != nil {
		panic(err)
	}
	getLines := func() []string {
		var output string
		for {
			line, err := r.ReadString(':')
			if err != nil {
				panic(err)
			}
			fmt.Println("[yolo] [stdout] " + strings.TrimSpace(line))
			output += line
			if strings.Contains(line, "Enter") {
				break
			}
		}
		return strings.Split(output, "\n")
	}
	parseLines := func(lines []string) []common.Rectangle {
		var rects []common.Rectangle
		for i := 0; i < len(lines); i++ {
			if !strings.Contains(lines[i], "%") {
				continue
			}
			//parts := strings.Split(lines[i], ": ")
			//box.Class = parts[0]
			//box.Confidence, _ = strconv.Atoi(strings.Trim(parts[1], "%"))
			for !strings.Contains(lines[i], "Bounding Box:") {
				i++
			}
			parts := strings.Split(strings.Split(lines[i], ": ")[1], ", ")
			if len(parts) != 4 {
				panic(fmt.Errorf("bad bbox line %s", lines[i]))
			}
			var left, top, right, bottom int
			for _, part := range parts {
				kvsplit := strings.Split(part, "=")
				k := kvsplit[0]
				v, _ := strconv.Atoi(kvsplit[1])
				if k == "Left" {
					left = v
				} else if k == "Top" {
					top = v
				} else if k == "Right" {
					right = v
				} else if k == "Bottom" {
					bottom = v
				}
			}
			rects = append(rects, common.Rectangle{
				common.Point{float64(left), float64(top)},
				common.Point{float64(right), float64(bottom)},
			})
		}
		return rects
	}
	detections := [][]Detection{}
	saveRects := func(frameIdx int, rects []common.Rectangle) {
		for len(detections) <= frameIdx {
			detections = append(detections, []Detection{})
		}
		for _, rect := range rects {
			var points [][2]int
			for _, p := range rect.ToPolygon() {
				points = append(points, [2]int{int(p.X), int(p.Y)})
			}
			detections[frameIdx] = append(detections[frameIdx], Detection{
				Points: points,
			})
		}
	}
	getFrameIdx := func(fname string) int {
		parts := strings.Split(fname, ".jpg")
		frameIdx, err := strconv.Atoi(parts[0])
		if err != nil {
			panic(err)
		}
		return frameIdx
	}
	sort.Slice(files, func(i, j int) bool {
		fname1 := files[i].Name()
		fname2 := files[j].Name()
		return getFrameIdx(fname1) < getFrameIdx(fname2)
	})
	var prevFrameIdx int = -1
	for _, fi := range files {
		frameIdx := getFrameIdx(fi.Name())
		fmt.Printf("[yolo] processing %s (%d)\n", fi.Name(), frameIdx)
		lines := getLines()
		if prevFrameIdx != -1 {
			rects := parseLines(lines)
			saveRects(prevFrameIdx, rects)
		}
		stdin.Write([]byte(filepath.Join(videoPath, fi.Name()) + "\n"))
		prevFrameIdx = frameIdx
	}
	if prevFrameIdx != -1 {
		lines := getLines()
		rects := parseLines(lines)
		saveRects(prevFrameIdx, rects)
	}

	bytes, err := json.Marshal(detections)
	if err != nil {
		panic(err)
	}
	if err := ioutil.WriteFile(outPath, bytes, 0644); err != nil {
		panic(err)
	}
}

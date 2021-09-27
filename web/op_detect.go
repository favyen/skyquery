package main

import (
	"github.com/mitroadmaps/gomapinfer/common"

	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

type Detection struct {
	Points [][2]int
	OrigPoints [][2]int `json:",omitempty"`
}

func (d Detection) Polygon() common.Polygon {
	poly := common.Polygon{}
	for _, p := range d.Points {
		poly = append(poly, common.Point{float64(p[0]), float64(p[1])})
	}
	return poly
}

func PointsToPolyString(points [][2]int) string {
	var parts []string
	for _, point := range points {
		parts = append(parts, fmt.Sprintf("%v,%v", point[0], point[1]))
	}
	return strings.Join(parts, " ")
}

type DetectorConfig struct {
	// Either "yolov3" or "detector".
	Name string
	// TensorFlow checkpoint or YOLOv3 weights file.
	ModelPath string
	// If YOLOv3, the yolo.cfg path.
	ConfigPath string
	// If our detector, the input scale factor (from training).
	Resize float64
}

func DetectOp(args []OpArgument, outDir string) error {
	// Load the object detector configuration from the data directory.
	detectorName := args[0].String
	bytes, err := ioutil.ReadFile(filepath.Join(Config.DataDir, "detect", detectorName+".json"))
	if err != nil {
		return fmt.Errorf("error loading config for detector %s: %v", detectorName, err)
	}
	var detectorCfg DetectorConfig
	if err := json.Unmarshal(bytes, &detectorCfg); err != nil {
		return err
	}

	// Run the detector to get pixel coordinates of detections.
	rawDir := outDir+".raw"
	if detectorCfg.Name == "detector" {
		siftDir := outDir+".sift"
		// Align pairs of consecutive frames with each other using SIFT features.
		log.Printf("[op_detect] SIFT alignment")
		err := RunIfNeeded(siftDir, func(outDir string) error {
			cmd := exec.Command(Config.Python, "detector/sift_match.py", Config.VideoDir, outDir)
			cmd.Stdout = os.Stdout
			cmd.Stderr = os.Stderr
			return cmd.Run()
		})
		if err != nil {
			return err
		}

		// Run detector.
		log.Printf("[op_detect] inference")
		err = RunIfNeeded(rawDir, func(outDir string) error {
			cmd := exec.Command(Config.Python, "detector/infer.py", detectorCfg.ModelPath, siftDir, filepath.Join(outDir, "detect.json"))
			cmd.Stdout = os.Stdout
			cmd.Stderr = os.Stderr
			return cmd.Run()
		})
		if err != nil {
			return err
		}
	} else if detectorCfg.Name == "yolov3" {
		log.Printf("[op_detect] yolov3 inference")
		err = RunIfNeeded(rawDir, func(outDir string) error {
			cmd := exec.Command("go", "run", "yolov3/infer.go", detectorCfg.ConfigPath, detectorCfg.ModelPath, Config.VideoDir, filepath.Join(outDir, "detect.json"))
			cmd.Stdout = os.Stdout
			cmd.Stderr = os.Stderr
			return cmd.Run()
		})
		if err != nil {
			return err
		}
	} else {
		return fmt.Errorf("unknown detector name %s", detectorCfg.Name)
	}

	// Transform pixel coordinates to world coordinates.
	log.Printf("[op_detect] apply bounds")
	cmd := exec.Command(
		Config.Python, "detector/apply_bounds.py",
		filepath.Join(Config.DataDir, "align-out.json"),
		filepath.Join(rawDir, "detect.json"),
		filepath.Join(outDir, "detect.json"),
	)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	err = cmd.Run()
	if err != nil {
		return err
	}
	log.Printf("[op_detect] done")
	return nil
}

func init() {
	Ops["Detect"] = DetectOp
}

package main

import (
	"os"
)

type OpArgument struct {
	// Either "node" or "string"
	Type string

	String string

	// For "node" type, specifies folder on disk containing node's outputs.
	DirName string
}

var Ops = map[string]func(args []OpArgument, outDir string) error{}

// Run a function, but only if the outDir is not created yet.
// If it isn't there yet, we actually run the function to produce outputs in a temporary directory.
// Then we atomically rename the temporary directory to outDir.
func RunIfNeeded(outDir string, f func(string) error) error {
	if _, err := os.Stat(outDir); err == nil {
		return nil
	}

	tmpDir := outDir + ".tmp"
	// Delete the tmpDir in case it already exists.
	if err := os.RemoveAll(tmpDir); err != nil {
		return err
	}
	os.MkdirAll(tmpDir, 0755)
	// Run the function.
	if err := f(tmpDir); err != nil {
		return err
	}
	// Since function completed successfully, we can rename the tmpDir.
	if err := os.Rename(tmpDir, outDir); err != nil {
		return err
	}
	return nil
}

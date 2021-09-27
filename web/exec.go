package main

import (
	"log"
	"path/filepath"
)

func (g Graph) Exec() (map[string]string, error) {
	hashes := g.GetHashStrings()
	// Compute desired output directories of each node.
	outDirs := map[string]string{}
	for name, hash := range hashes {
		outDirs[name] = filepath.Join(Config.DataDir, g[name].Operation + "." + hash)
	}
	// Whether the outputs of each node in the graph have been computed.
	ready := map[string]bool{}
	// Repeatedly loop over nodes in the graph until all nodes are ready.
	for len(ready) < len(g) {
		for name, node := range g {
			if ready[name] {
				continue
			}

			// Are all parents ready?
			okay := true
			for _, parent := range node.Parents() {
				if !ready[parent.Name] {
					okay = false
				}
			}
			if !okay {
				continue
			}

			// Run the node (if needed).
			err := RunIfNeeded(outDirs[name], func(outDir string) error {
				var arguments []OpArgument
				for _, arg := range node.Arguments {
					if arg.Type == "string" {
						arguments = append(arguments, OpArgument{
							Type: "string",
							String: arg.String,
						})
					} else if arg.Type == "node" {
						arguments = append(arguments, OpArgument{
							Type: "node",
							DirName: outDirs[arg.Node.Name],
						})
					}
				}

				log.Printf("[node %s] run operation %s", name, node.Operation)
				return Ops[node.Operation](arguments, outDir)
			})
			if err != nil {
				return nil, err
			}
			ready[name] = true
		}
	}

	return outDirs, nil
}

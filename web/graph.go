package main

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
)

type Argument struct {
	// Either "node" or "string"
	Type string

	String string
	Node *Node
}

type Node struct {
	Name string
	Operation string
	Arguments []Argument
}

func (node *Node) Parents() []*Node {
	parents := []*Node{}
	for _, argument := range node.Arguments {
		if argument.Type != "node" {
			continue
		}
		parents = append(parents, argument.Node)
	}
	return parents
}

type Graph map[string]*Node

// Returns hashes of all nodes in the graph.
func (graph Graph) GetHashes() map[string][]byte {
	hashes := make(map[string][]byte)
	// repeatedly iterate over graph and add hashes for nodes
	// where parent hashes are already computed
	for len(hashes) < len(graph) {
		for name, node := range graph {
			if hashes[name] != nil {
				continue
			}
			// collect parent hashes
			missing := false
			for _, parent := range node.Parents() {
				if hashes[parent.Name] == nil {
					missing = true
					break
				}
			}
			if missing {
				continue
			}

			h := sha256.New()
			for _, arg := range node.Arguments {
				if arg.Type == "string" {
					h.Write([]byte(fmt.Sprintf("%s\n", arg.String)))
				} else if arg.Type == "node" {
					hash := hashes[arg.Node.Name]
					h.Write([]byte(fmt.Sprintf("%s\n", string(hash))))
				}
			}
			h.Write([]byte(fmt.Sprintf("%s", node.Operation)))
			hashes[name] = h.Sum(nil)
		}
	}
	return hashes
}

func (graph Graph) GetHashStrings() map[string]string {
	hashes := make(map[string]string)
	for name, bytes := range graph.GetHashes() {
		hashes[name] = hex.EncodeToString(bytes)
	}
	return hashes
}

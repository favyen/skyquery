package main

import (
	"fmt"
	"log"
	"strings"
)

func ParseQuery(query string) (Graph, error) {
	lines := strings.Split(query, "\n")
	graph := Graph{}
	for i, line := range lines {
		line = strings.TrimSpace(line)
		log.Println(query, line, lines)
		if line == "" {
			continue
		}

		// line like newtable = Op(node_arg, "string_arg", ...)
		eqParts := strings.Split(line, "=")
		nodeName := strings.TrimSpace(eqParts[0])
		rhs := strings.TrimSpace(eqParts[1])
		opName := strings.Split(rhs, "(")[0]
		argStr := strings.Split(strings.Split(rhs, "(")[1], ")")[0]
		argParts := strings.Split(argStr, ";")
		var arguments []Argument
		for _, argPart := range argParts {
			argPart = strings.TrimSpace(argPart)
			if argPart[0] == '"' {
				// string argument
				arguments = append(arguments, Argument{
					Type: "string",
					String: argPart[1:len(argPart)-1],
				})
			} else {
				// node argument
				refName := argPart
				if graph[refName] == nil {
					return nil, fmt.Errorf("line %d: referenced variable %s not defined yet", i+1, refName)
				}
				arguments = append(arguments, Argument{
					Type: "node",
					Node: graph[refName],
				})
			}
		}
		node := &Node{
			Name: nodeName,
			Operation: opName,
			Arguments: arguments,
		}
		log.Printf("[parse] [%s] => add node name %s op %s with %d arguments", line, node.Name, node.Operation, len(node.Arguments))
		graph[nodeName] = node
	}
	return graph, nil
}

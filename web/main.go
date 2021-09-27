package main

import (
	"encoding/json"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"sync"
)

func main() {
	Config.DataDir = os.Args[1]
	Config.VideoDir = os.Args[2]
	Config.Python = "python3.6"

	var mu sync.Mutex
	var running bool

	fileServer := http.FileServer(http.Dir("web/static/"))
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/" {
			w.Header().Set("Cache-Control", "no-cache")
		}
		fileServer.ServeHTTP(w, r)
	})
	http.HandleFunc("/exec", func(w http.ResponseWriter, r *http.Request) {
		r.ParseForm()
		query := r.PostForm.Get("query")
		mu.Lock()
		defer mu.Unlock()
		if running {
			http.Error(w, "another query is already running", 400)
			return
		}
		running = true
		go func() {
			defer func() {
				mu.Lock()
				running = false
				mu.Unlock()
			}()

			// Parse and execute query.
			log.Printf("[main] parsing query")
			graph, err := ParseQuery(query)
			if err != nil {
				log.Printf("error parsing query: %v", err)
				return
			}
			log.Printf("[main] executing query")
			outDirs, err := graph.Exec()
			if err != nil {
				log.Printf("error running query: %v", err)
				return
			}
			log.Printf("[main] execution completed")

			// Compute visualization of the "out" table (if set).
			if graph["out"] == nil {
				return
			}
			log.Printf("[main] visualizing output table")
			err = Visualize(outDirs["out"])
			if err != nil {
				log.Printf("error visualizing output table: %v", err)
				return
			}
			log.Printf("[main] visualization ready")
		}()
	})
	http.HandleFunc("/state", func(w http.ResponseWriter, r *http.Request) {
		var state struct {
			Running bool
		}
		mu.Lock()
		state.Running = running
		mu.Unlock()
		jsonResponse(w, state)
	})
	http.HandleFunc("/vis", func(w http.ResponseWriter, r *http.Request) {
		visPath := filepath.Join(Config.DataDir, "out.jpg")
		if _, err := os.Stat(visPath); err != nil {
			http.Error(w, "not found", 404)
			return
		}
		http.ServeFile(w, r, visPath)
	})
	log.Printf("starting on :8080")
	log.Fatal(http.ListenAndServe(":8080", nil))
}

func jsonResponse(w http.ResponseWriter, x interface{}) {
	bytes, err := json.Marshal(x)
	if err != nil {
		panic(err)
	}
	w.Header().Set("Content-Type", "application/json")
	w.Write(bytes)
}

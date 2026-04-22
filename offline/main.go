package main

import (
	"fmt"
	"log"
	"net/http"
)

func main() {
	// Serve static files from the current directory
	fs := http.FileServer(http.Dir("."))
	http.Handle("/", fs)

	port := ":6977"
	fmt.Printf("Server started at http://localhost%s\n", port)
	log.Fatal(http.ListenAndServe(port, nil))
}
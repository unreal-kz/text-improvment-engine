package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"strings"
)

func main() {
	// Define flags
	textInput := flag.String("text", "", "Text to analyze")
	fileInput := flag.String("file", "", "File containing text to analyze")

	// Parse the flags
	flag.Parse()

	var text string

	// Check if text is provided directly
	if *textInput != "" {
		text = *textInput
	} else if *fileInput != "" {
		// Read from file
		file, err := os.Open(*fileInput)
		if err != nil {
			fmt.Println("Error opening file:", err)
			return
		}
		defer file.Close()

		scanner := bufio.NewScanner(file)
		var fileText strings.Builder
		for scanner.Scan() {
			fileText.WriteString(scanner.Text() + "\n")
		}

		if err := scanner.Err(); err != nil {
			fmt.Println("Error reading from file:", err)
			return
		}

		text = fileText.String()
	} else {
		fmt.Println("Please provide text or a file path.")
		return
	}

	// Process the text
	fmt.Println("Processing text:", text)
	// Here, you will later add the code to process and analyze the text
}

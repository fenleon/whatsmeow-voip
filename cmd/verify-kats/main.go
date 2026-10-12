// Command verify-kats runs the crypto known-answer vectors from
// vectors/voip/kats.json against the wavoip package and exits non-zero on any
// mismatch. Run from the repo root (or pass the JSON path as an argument).
package main

import (
	"fmt"
	"os"

	"github.com/fenleon/whatsmeow-voip"
)

func main() {
	path := "vectors/voip/kats.json"
	if len(os.Args) > 1 {
		path = os.Args[1]
	}
	data, err := os.ReadFile(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "verify-kats: %v\n", err)
		os.Exit(2)
	}
	if err := wavoip.VerifyKats(data); err != nil {
		fmt.Fprintf(os.Stderr, "verify-kats: FAIL\n%v\n", err)
		os.Exit(1)
	}
	fmt.Println("verify-kats: all known-answer vectors pass")
}

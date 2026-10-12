package wavoip

import (
	"os"
	"testing"

	"github.com/stretchr/testify/require"
)

// The vectors live in vectors/voip/kats.json relative to the repo root.
func loadKats(t *testing.T) []byte {
	t.Helper()
	for _, path := range []string{"vectors/voip/kats.json", "../vectors/voip/kats.json"} {
		data, err := os.ReadFile(path)
		if err == nil {
			return data
		}
	}
	t.Fatal("kats.json not found (run from the repo root)")
	return nil
}

func TestVerifyKats(t *testing.T) {
	data := loadKats(t)
	t.Run("media crypto stack matches reference vectors", func(t *testing.T) {
		require.NoError(t, VerifyKats(data))
	})
}

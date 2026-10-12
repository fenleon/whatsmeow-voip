# Go style — Beeper/mautrix guidelines (applied to this repo)

Distilled from the [Beeper Go Guidelines](https://beeper.notion.site/Beeper-Go-Guidelines-ae943532d96f4ad6a614baf836c073eb)
and the [mautrix contributing guidelines](https://docs.mau.fi/bridges/general/contributing.html).
These bind every Go commit in this repo. Base guides: [Effective Go](https://go.dev/doc/effective_go)
and [CodeReviewComments](https://go.dev/wiki/CodeReviewComments).

## Upstream process (mautrix contributing)

- Non-trivial changes: discuss the plan in the project's Matrix room **before**
  opening a PR; PRs may be rewritten rather than reviewed.
- Issues, PR descriptions, comments: written by the human author. AI may help
  with code; AI slop is not accepted and the author is accountable for every
  line (and must have tested it).
- `pre-commit install` before committing; `pre-commit run -a` before pushing.
  Config in `.pre-commit-config.yaml` (whatsmeow's own baseline: go-imports
  with `-local go.mau.fi/whatsmeow`, whitespace/EOF/yaml hooks).
- Latest Go or the previous release only.
- No changelog edits in PRs (maintainer does releases).

## Style

- `any` over `interface{}`; lowerCamelCase variables; consistent initialism
  casing (`URL`, `JID`, `ID`).
- `var x T` over `x := T{}` for zero values.
- Indent error flow: handle the error in a small block and return; left-aligned
  code reads as the happy path. Exception: a short `if/else if` chain of error
  checks is fine.
- `http.StatusX` constants, never raw status codes.
- Slices over maps wherever iteration order could matter.

## Errors & logging

- Wrap errors with context: `fmt.Errorf("failed to X: %w", err)`.
- Libraries (what this repo starts as) **never configure loggers** — accept
  `*zerolog.Logger` parameters or pull from `context.Context` via
  `zerolog.Ctx(ctx)`; configuring belongs to the top-level program.
- Structured logging: no `Msgf` for values — use typed fields; log keys
  `snake_case`; always attach the error with `.Err(err)` where relevant.
- Every `if err != nil` either logs or returns to a caller that logs.
- Levels: `fatal` = process must die (config invalid, required auth failed);
  `error` = recovered but a bug (permanent failure of a should-always-succeed
  request, bad DB query); `warn` = handled gracefully (malformed remote data,
  non-essential task failed); `info` = lifecycle milestones; `debug` = control
  flow explanations ("skipping X because Y"); `trace` = verbose state tracking,
  the **only** level allowed to carry sensitive content — must be off in
  production. Never `panic` level.

## Testing

- `testify` (`require`/`assert`).
- Table-driven tests with `t.Run` subtests, liberally.

## Databases (if/when needed)

- Constant SQL strings over sprintf'd queries; `NOT NULL` fields by default;
  `QueryRow` for single rows; bubble errors up (only `sql.ErrNoRows` may
  become a nil value with callers handling nil).

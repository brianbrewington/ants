# CI

`ci.yml` is the GitHub Actions workflow (ruff + pytest for the backend; typecheck
+ build for the frontend, on every push / PR).

It lives here instead of `.github/workflows/` because the personal access token
used to push the repo doesn't carry the `workflow` scope (GitHub blocks pushing
workflow files without it). **To activate CI**, do either:

- copy it into place and push with a `workflow`-scoped token:
  ```bash
  mkdir -p .github/workflows && cp ci/ci.yml .github/workflows/ci.yml
  git add .github/workflows/ci.yml && git commit -m "Enable CI" && git push
  ```
- or paste the contents into a new workflow via the GitHub web UI
  (Actions → New workflow), which doesn't need the extra scope.

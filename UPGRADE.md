# Updating an existing checkout to 0.3.0

Run these in the codespace terminal, one line at a time, after uploading
`GuidedVina-ACO-0.3.0.tar.gz` to the repository through github.com.

```bash
git pull
tar -xzf GuidedVina-ACO-0.3.0.tar.gz
rm -f GuidedVina-ACO-0.3.0.tar.gz
shopt -s dotglob
cp -r GuidedVina-ACO/* .
rm -rf GuidedVina-ACO
python3 -m pytest -q          # expect: 70 passed
git add -A
git commit -m "Bias through affinity maps (0.3.0)"
git push
```

## What changed

Vina-ACO has no `--bias` option, so bias is now applied to the affinity maps
instead. See the Bias section of `docs/METHOD.md`.

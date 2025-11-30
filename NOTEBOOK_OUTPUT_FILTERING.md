# Notebook Output Filtering

## Overview

This repository uses **nbstripout** to automatically strip outputs from Jupyter notebooks before they are committed to git. This keeps the repository clean and prevents:

- Large binary blobs from plots/images
- Merge conflicts in output cells
- Unnecessary repository bloat
- Potential data leaks in outputs

## Setup

The filtering is configured via:

1. **Git filter** (configured by `nbstripout --install`):
   ```
   filter.nbstripout.clean = python -m nbstripout
   filter.nbstripout.smudge = cat
   filter.nbstripout.required = true
   ```

2. **.gitattributes** file:
   ```
   *.ipynb filter=nbstripout
   ```

This means:
- When you commit: outputs are stripped
- When you checkout: notebooks are left as-is
- If the filter fails: commit is prevented

## Verification

The setup was verified with these tests:

### Test 1: nbstripout works
```bash
# Created test notebook with outputs
nbstripout /tmp/test_notebook.ipynb
# Verified: outputs stripped ✅
```

### Test 2: Git filter works
```bash
# Tested git hash-object with filter
git hash-object -w --stdin --path=test.ipynb < notebook_with_outputs.ipynb
# Verified: filter applied ✅
```

### Test 3: Repository notebooks
```bash
# Checked all notebooks in repo
python -c "import json; ..."
# Result: 0 cells with outputs ✅
```

## Usage

### For Contributors

**No action needed!** The filter runs automatically:

```bash
# Edit notebook, run cells, save with outputs
jupyter notebook examples/my-notebook.ipynb

# Commit normally - outputs are stripped automatically
git add examples/my-notebook.ipynb
git commit -m "Add new analysis notebook"

# Outputs are stripped in the commit, but your local file keeps them
```

### Installing nbstripout (for new clones)

If you clone the repository on a new machine:

```bash
# The filter is already configured in .git/config after clone
# Just ensure nbstripout is installed:
pip install nbstripout

# Or install from requirements.txt which includes it
pip install -r requirements.txt
```

The git filter configuration is stored in `.git/config` and will be set up when you run `nbstripout --install` (already done for this repository).

### Checking Notebook Status

To see if a notebook has outputs that will be stripped:

```bash
# Check a specific notebook
python -c "
import json
with open('examples/my-notebook.ipynb') as f:
    nb = json.load(f)
outputs = sum(1 for c in nb['cells'] if c.get('outputs'))
print(f'Cells with outputs: {outputs}')
"
```

### Manual Stripping (if needed)

```bash
# Strip outputs from a notebook manually
nbstripout examples/my-notebook.ipynb

# Strip outputs from all notebooks
find . -name '*.ipynb' -not -path '*/.ipynb_checkpoints/*' -exec nbstripout {} \;
```

## Testing the Filter

To verify the filter is working:

```bash
# 1. Create a test notebook with outputs
jupyter notebook /tmp/test.ipynb
# (run some cells)

# 2. Try to add and see what would be committed
git add /tmp/test.ipynb
git diff --cached /tmp/test.ipynb

# The diff should show outputs are stripped
```

## Configuration Files

### .gitattributes
```
*.ipynb filter=nbstripout
```

### .git/config (automatically created)
```
[filter "nbstripout"]
    clean = python -m nbstripout
    smudge = cat
    required = true
```

### .gitignore (already configured)
```
.ipynb_checkpoints/
```

## Benefits

1. **Smaller repository**: No binary output data committed
2. **Cleaner diffs**: Only source code changes visible
3. **No merge conflicts**: Outputs can't conflict
4. **Faster clones**: Less data to download
5. **Privacy**: No accidental data leaks in outputs

## Troubleshooting

### Filter not working?

Check the configuration:
```bash
git config --local filter.nbstripout.clean
# Should output: python -m nbstripout
```

### nbstripout not found?

Install it:
```bash
pip install nbstripout
```

### Want to commit outputs (unusual)?

Temporarily disable the filter:
```bash
git -c filter.nbstripout.clean= add notebook.ipynb
```

## Related Files

- `.gitattributes` - Applies filter to *.ipynb files
- `tests/test_notebook.py` - Notebook testing infrastructure
- `TESTING.md` - Testing documentation
- `requirements.txt` - Includes nbstripout dependency

## References

- [nbstripout documentation](https://github.com/kynan/nbstripout)
- [Git filters documentation](https://git-scm.com/book/en/v2/Customizing-Git-Git-Attributes)


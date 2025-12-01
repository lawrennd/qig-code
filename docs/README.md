# Read the Docs Setup

This document describes how to connect the qig repository to Read the Docs for automatic documentation building.

## Prerequisites

- Repository pushed to GitHub/GitLab/Bitbucket
- `.readthedocs.yaml` configuration file in repository root (✅ already created)
- `docs/requirements.txt` with build dependencies (✅ already created)

## Setup Steps

### 1. Create Read the Docs Account

1. Go to https://readthedocs.org/
2. Sign up or log in with your GitHub/GitLab/Bitbucket account

### 2. Import Project

1. Click "Import a Project"
2. Select your qig repository from the list
3. Fill in project details:
   - **Name**: `qig` (or `quantum-inaccessible-game`)
   - **Repository URL**: Your repository URL
   - **Default branch**: `main` (or your default branch)
   - **Language**: Python

### 3. Configure Build

The build is already configured via `.readthedocs.yaml`. No additional configuration needed!

The file specifies:
- Python 3.11
- Ubuntu 22.04
- Sphinx documentation builder
- Dependencies from `docs/requirements.txt`

### 4. Trigger First Build

1. Go to your project dashboard on Read the Docs
2. Click "Build Version"
3. Select "latest" or your default branch
4. Wait for build to complete

### 5. Verify Documentation

Once the build completes:
1. Visit `https://qig.readthedocs.io/` (or your project URL)
2. Verify all sections render correctly:
   - Getting Started (installation, quickstart)
   - User Guide
   - API Reference
   - Theory
   - Development Guide

## Troubleshooting

### Build Fails with Import Errors

If the build fails because it can't import `qig` modules:
- Check that `docs/requirements.txt` includes numpy, scipy, matplotlib
- Verify `.readthedocs.yaml` has `path: .` under python install

### Missing Extensions

If Sphinx extensions are not found:
- Check that all extensions in `docs/source/conf.py` are listed in `docs/requirements.txt`
- Common extensions: sphinx, sphinx-rtd-theme, sphinx-autodoc-typehints, nbsphinx

### Warnings in Build

Some warnings are expected:
- Docstring formatting warnings (can be fixed incrementally)
- Missing cross-references (will improve as documentation expands)

Set `fail_on_warning: false` in `.readthedocs.yaml` to allow builds with warnings.

## Automation

Once connected, Read the Docs will automatically:
- Build documentation on every push to the repository
- Build documentation for pull requests (if configured)
- Host multiple versions (latest, stable, by tag)

## Custom Domain (Optional)

To use a custom domain like `docs.yourdomain.com`:
1. Go to Admin → Domains in Read the Docs
2. Add your custom domain
3. Configure DNS CNAME record as instructed

## Further Configuration

See:
- [Read the Docs Documentation](https://docs.readthedocs.io/)
- [.readthedocs.yaml Reference](https://docs.readthedocs.io/en/stable/config-file/v2.html)
- [Sphinx Documentation](https://www.sphinx-doc.org/)


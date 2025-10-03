# Pre-commit Migration Guide

## 🚨 Important: Pre-commit hooks are currently DISABLED by default

Pre-commit hooks will reformat ALL files in the repository, which can cause merge conflicts with existing branches.

## Migration Timeline

### Phase 1: Preparation (Current)
- ✅ Pre-commit configuration is ready but disabled by default
- ✅ Deploy script uses fallback formatting
- ✅ No impact on existing branches
- ✅ Individual developers can opt-in with `--force` flag

### Phase 2: Migration Day (After all branches merge)
- Enable pre-commit hooks project-wide
- Run formatting on entire codebase once
- All future commits use consistent formatting

### Phase 3: Enforcement (Ongoing)
- All new branches automatically formatted
- Consistent code style across project

## Current Behavior

### Default (Safe Mode)
```bash
./setup_dev.sh --install
# Shows warning and asks for confirmation
# Recommended: Choose "No" until migration day
```

### Force Mode (Advanced Users Only)
```bash
./setup_dev.sh --install --force
# Skips warnings and installs immediately
# ⚠️ May cause merge conflicts with existing branches
```

### Check Status
```bash
./setup_dev.sh --check
# Shows current pre-commit setup status
```

## Migration Day Commands

**🎯 When ready to migrate the entire project:**

```bash
# 1. Ensure all branches are merged
git checkout main
git pull origin main

# 2. Install pre-commit hooks (force mode)
./setup_dev.sh --install --force

# 3. Format entire codebase
poetry run pre-commit run --all-files

# 4. Commit the formatting changes
git add .
git commit -m "Apply consistent formatting project-wide

- Enable pre-commit hooks
- Format all files with Black and Ruff
- All future commits will maintain consistent style

Closes #XXX"

# 5. Push and notify team
git push origin main
```

## Post-Migration

After migration day, all developers should run:

```bash
git pull origin main
./setup_dev.sh --install --force
```

## Manual Formatting (Current Fallback)

Until migration day, you can manually format your code:

```bash
# Quick format everything
./format.sh

# Individual tools
poetry run black src/ tests/
poetry run ruff check src/ tests/ --fix
poetry run ruff format src/ tests/
```

## Deploy Script Behavior

The deploy script automatically detects pre-commit setup:

- **If pre-commit installed**: Uses pre-commit hooks
- **If not installed**: Falls back to manual formatting
- **No impact on deployment process**

## FAQ

### Q: Why not just enable pre-commit hooks now?
A: It would reformat ALL files, causing merge conflicts for active branches.

### Q: Can I use pre-commit on my feature branch?
A: Yes, use `--force` flag, but expect merge conflicts when merging to main.

### Q: When will the migration happen?
A: After all current active branches are merged (planned for next week).

### Q: What if I forget to install pre-commit after migration?
A: The deploy script will catch formatting issues and apply them automatically.

### Q: Will this change my development workflow?
A: Minimally. Code gets formatted automatically on commit. You can skip with `git commit --no-verify` if needed.

## Troubleshooting

### Pre-commit hooks fail
```bash
# Re-install hooks
./setup_dev.sh --install --force

# Update hook repositories
poetry run pre-commit autoupdate
```

### Formatting issues
```bash
# Manual format to see what's happening
./format.sh

# Check what Ruff wants to fix
poetry run ruff check src/ tests/
```

### Import errors after adding new code
```bash
# Update lazy imports
poetry run mkinit --lazy_loader src/crantpy --recursive --inplace
```

## Timeline Summary

| Phase | Status | Action |
|-------|---------|---------|
| **Now** | ⏳ Preparation | Pre-commit ready but disabled |
| **Next Week** | 🚀 Migration Day | Enable project-wide formatting |
| **Ongoing** | ✅ Enforcement | Consistent formatting for all |

---

*This migration strategy ensures zero disruption to current development while preparing for consistent code formatting across the entire project.*

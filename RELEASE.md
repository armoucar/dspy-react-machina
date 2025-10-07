# Release Process

1. Update version in `pyproject.toml`
2. Clean and build: `rm -rf dist/* && uv build`
3. Publish: `uv publish --username __token__ --password <token-from-~/.pypirc>`

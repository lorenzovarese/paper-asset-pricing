# paper-asset-pricing: P.A.P.E.R.
Platform for Asset Pricing Experimentation and Research

### Publishing Workflow

1.  **Ensure LICENSE files:** Place a `LICENSE` file in each of the sub-project directories (`paper-data/LICENSE`, `paper-model/LICENSE`, etc.).
2.  **Build each package:**
    *   `cd paper-data && uv build` (or `python -m build`)
    *   `cd ../paper-model && uv build`
    *   `cd ../paper-portfolio && uv build`
    *   `cd ../paper-tools && uv build`
    Each command will create `.whl` and `.tar.gz` files in its respective `dist/` directory.
3.  **Publish to PyPI (order matters for dependencies):**
    *   Use `twine` for uploading. First time, you might want to use TestPyPI.
    *   `twine upload paper-data/dist/*`
    *   `twine upload paper-model/dist/*`
    *   `twine upload paper-portfolio/dist/*` (depends on paper-data, paper-model)
    *   `twine upload paper-tools/dist/*` (depends on the others via extras)

This setup allows users to:

*   `pip install paper-tools` (installs the minimal suite package)
*   `pip install paper-tools[data]` (installs `paper-tools` and `paper-data`, making `paper-data` CLI available)
*   `pip install paper-tools[models]` (installs `paper-tools` and `paper-model`, making its CLI available)
*   `pip install paper-tools[portfolio]` (installs `paper-tools` and `paper-portfolio`, which in turn pulls `paper-data` and `paper-model`. `paper-portfolio`'s CLI becomes available).
*   `pip install paper-data` (installs just the data tools)

This structure is robust and scalable for your project. Remember to manage version numbers carefully, especially in the dependency specifications (e.g., `paper-data~=0.1.0`).
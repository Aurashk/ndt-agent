## GitHub Copilot Instructions for Ultrasonic NDT Project

This project post-processes ultrasonic non-destructive testing (NDT) measurement data to generate C-scan data.

- For every contribution you should create your own branch from `main` and submit a pull request with a clear description of the changes. Run and test in your environment first.
    
- For coding standards and styling, refer to [docs/codingstyle.md](../docs/codingstyle.md).

- For debugging guidelines, see [docs/debugging.md](../docs/debugging.md).

- For testing requirements see [docs/testing.md](../docs/testing.md).
All new functions need corresponding pytest tests in the tests/ directory.
 Run tests with `pytest` and aim for >90% coverage on new code. Include both unit tests and integration tests with realistic test data.

- Follow the established pattern for file naming and organisation: test files use `test_*.py`, debug images get auto-incremented names like `001_signal_analysis.png`, and documentation goes in the `docs/` directory with descriptive names.

- Check requirements.txt for dependencies. If you add or remove a new you should run `pip freeze | findstr /V "cscanmaker" > requirements.txt` to update it.

- When creating or updating documentation, reference the existing comprehensive docs in README.md, codingstyle.md, debugging.md, testing.md. Keep examples practical and related to ultrasonic NDT applications like flaw detection and material characterization.
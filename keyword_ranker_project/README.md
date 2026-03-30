# Keyword Ranker Project

This project ranks a known list of keywords for a target document using a simple, memory-conscious TF-IDF approach.

## Files

- `app.py` - main entry point
- `keyword_loader.py` - loads keywords from a separate file
- `tfidf_ranker.py` - streaming TF-IDF keyword ranking logic
- `topn_ranker.py` - returns the top N ranked keywords
- `tests/` - pytest test suite
- `run_pytest.sh` - shell script to install requirements and run tests
- `run_pytest.bat` - Windows batch script to install requirements and run tests

## Run the app

```bash
python app.py --keywords keywords.txt --document sample_document.txt --corpus-dir corpus --top-n 5
```

## Run tests on Linux/macOS

```bash
./run_pytest.sh
```

## Run tests on Windows

```bat
run_pytest.bat
```

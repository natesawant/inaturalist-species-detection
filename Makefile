venv:
	rm -rf .venv
	python -m venv .venv
	sh .venv/bin/activate
	pip install -r requirements.txt
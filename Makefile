venv:
	if [ -d ".venv" ]; then rm -rf .venv; fi
	python -m venv .venv
	sh .venv/bin/activate
	pip install -r requirements.txt

py3venv:
	if [ -d ".venv" ]; then rm -rf .venv; fi
	python3.11 -m venv .venv
	sh .venv/bin/activate
	pip install -r requirements.txt
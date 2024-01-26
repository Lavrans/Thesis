VENV = .venv
REQ = requirements.txt

.PHONY: install clean freeze remove_venv mypy ruff-lint lint ruff-fmt sort-imports fmt

install: venv
	$(VENV)/bin/pip install -r $(REQ)

venv:
	python3 -m venv $(VENV)

remove_venv:
	rm -r $(VENV)

clean: remove_venv venv install

freeze:
	$(VENV)/bin/pip freeze > $(REQ)

mypy: 
	$(VENV)/bin/mypy ./code/

ruff-lint: 
	$(VENV)/bin/ruff check

lint: ruff-lint mypy

ruff-fmt:
	$(VENV)/bin/ruff format

sort-imports:
	$(VENV)/bin/ruff . --select I --fix

fmt: ruff-fmt sort-imports

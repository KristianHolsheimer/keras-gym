PYTHON_EXEC = python3

all: clean src wheel

clean:
	$(PYTHON_EXEC) setup.py clean
	rm -rf dist build *.egg-info

docs: clean_docs
	$(PYTHON_EXEC) setup.py build_sphinx

clean_docs:
	rm -rf build/sphinx doc/build

src:
	$(PYTHON_EXEC) setup.py sdist

wheel:
	$(PYTHON_EXEC) setup.py bdist_wheel

install:
	$(PYTHON_EXEC) setup.py install

flake8:
	$(PYTHON_EXEC) setup.py flake8

upload: all
	twine upload -u krispisvis dist/*

nbconvert:
	rm -f doc/_static/notebooks/*.html
	jupyter nbconvert --to html --output-dir doc/_static/notebooks/ notebooks/*.ipynb

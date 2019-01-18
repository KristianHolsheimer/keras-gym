PYTHON_EXEC = python3

all: clean src wheel

clean:
	$(PYTHON_EXEC) setup.py clean

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

PYTHON_EXEC = python3

all: clean src wheel

clean:
	$(PYTHON_EXEC) setup.py clean

docs:
	$(PYTHON_EXEC) setup.py build_sphinx

docs refresh:
	$(PYTHON_EXEC) setup.py build_sphinx --all-files

src:
	$(PYTHON_EXEC) setup.py sdist

wheel:
	$(PYTHON_EXEC) setup.py bdist_wheel

install:
	$(PYTHON_EXEC) setup.py install

flake8:
	$(PYTHON_EXEC) setup.py flake8

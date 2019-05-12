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
	$(PYTHON_EXEC) -m twine upload -u krispisvis dist/*

test:
	$(PYTHON_EXEC) -m pytest

nbconvert:
	rm -f doc/_static/notebooks/*/*.html doc/_static/notebooks/*.html
	for f in $(find notebooks/*/*.ipynb); do jupyter nbconvert --to html --output-dir doc/_static/$(dirname $f) $f; done

install_dev: install_requirements
	$(PYTHON_EXEC) -m pip install -e .

install_requirements:
	for r in requirements.txt requirements.dev.txt doc/requirements.txt; do $(PYTHON_EXEC) -m pip install -r $$r; done

upgrade_requirements:
	for r in requirements.txt requirements.dev.txt doc/requirements.txt; do $(PYTHON_EXEC) -m pur -r $$r; $(PYTHON_EXEC) -m pip install -r $$r; done

tf_gpu:
	$(PYTHON_EXEC) -m pip uninstall --yes tensorflow && $(PYTHON_EXEC) -m pip install tensorflow-gpu

tf_cpu:
	$(PYTHON_EXEC) -m pip uninstall --yes tensorflow-gpu && $(PYTHON_EXEC) -m pip install tensorflow

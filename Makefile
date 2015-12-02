.PHONY: all install clean test docs

all:
	python setup.py build

install:
	python setup.py install

clean:
	python setup.py clean
	rm -rf build
	rm -rf docs
	rm -rf .autodocs/.build
	rm -rf .autodocs/*.rst

test:
	nosetests

docs:
	sphinx-apidoc -o .autodocs zephyr
	$(MAKE) -C .autodocs html
	ln -s .autodocs/.build/html ./docs
    

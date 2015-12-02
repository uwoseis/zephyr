.PHONY: all install clean test tests docs pnggraphs dotgraphs graphs

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

tests:
	nosetests

test: tests

docs:
	rm -f docs
	sphinx-apidoc -o .autodocs zephyr
	$(MAKE) -C .autodocs html
	ln -s .autodocs/.build/html ./docs
    
pnggraphs:
	mkdir -p graphs && cd graphs && pyreverse -my -A -o png -p zephyr ../zephyr/**/**.py

dotgraphs:
	mkdir -p graphs && cd graphs && pyreverse -my -A -o dot -p zephyr ../zephyr/**/**.py

graphs: dotgraphs pnggraphs

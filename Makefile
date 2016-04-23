.PHONY: all install clean test tests docs pnggraphs epsgraphs svggraphs dotgraphs graphs publish testpublish

all:
	python setup.py build

install:
	python setup.py install

clean:
	python setup.py clean
	rm -rf build
	rm -rf zephyr.egg-info
	rm -rf docs
	rm -rf .autodocs/.build
	rm -rf .autodocs/*.rst
	rm -rf graphs

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

svggraphs:
	mkdir -p graphs && cd graphs && pyreverse -my -A -o svg -p zephyr ../zephyr/**/**.py

epsgraphs:
	mkdir -p graphs && cd graphs && pyreverse -my -A -o eps -p zephyr ../zephyr/**/**.py

dotgraphs:
	mkdir -p graphs && cd graphs && pyreverse -my -A -o dot -p zephyr ../zephyr/**/**.py

pdfgraphs:
	mkdir -p graphs && cd graphs && pyreverse -my -A -o pdf -p zephyr ../zephyr/**/**.py

graphs: dotgraphs pnggraphs

publish:
	python setup.py sdist upload -r pypi

testpublish:
	python setup.py sdist upload -r pypitest


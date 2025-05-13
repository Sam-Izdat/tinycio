# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = ./doc
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: deep_autograde

deep_autograde:
	$(SPHINXBUILD) -b latex -D master_doc=source/deep_autograde -t deepdive "$(SOURCEDIR)" "$(BUILDDIR)/deep_autograde"
	$(MAKE) -C "$(BUILDDIR)/deep_autograde"
	mkdir -p $(BUILDDIR)/html/articles
	cp "$(BUILDDIR)/deep_autograde/tinycio.pdf" "$(BUILDDIR)/html/articles/deep_autograde.pdf"
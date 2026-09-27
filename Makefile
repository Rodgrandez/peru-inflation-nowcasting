PY ?= python
.PHONY: data nowcast report all test lint
data:    ; $(PY) -m nowcast.pipeline data
nowcast: ; $(PY) -m nowcast.pipeline nowcast
report:  ; $(PY) -m nowcast.pipeline report
all:     ; $(PY) -m nowcast.pipeline all
test:    ; $(PY) -m pytest -q
lint:    ; ruff check src tests

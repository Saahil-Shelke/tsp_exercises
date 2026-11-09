# DIGIT tactile sensor - data loading, replay, frame extraction, illumination study
# Run `make help` to list the commands.

VENV    := .venv
PY      := . $(VENV)/bin/activate && python
DATA    := datasets
FILE    ?= $(DATA)/recording_cent.touch
SPEED   ?= 1.0
TOP     ?= 3
SERIAL  ?= D20758
FRAMES  ?= 30
SWEEP   ?= $(shell ls -t outputs/collected/*.h5 2>/dev/null | head -1)

.DEFAULT_GOAL := help
.PHONY: help setup info replay export export-all frames illumination \
        collect collect-rgb collect-free collect-sim analyze-sweep all clean clean-all

help: ## Show this help
	@awk 'BEGIN{FS=":.*## "} /^[a-zA-Z_-]+:.*## /{printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "  Variables: FILE=$(FILE) SPEED=$(SPEED) TOP=$(TOP) SERIAL=$(SERIAL) FRAMES=$(FRAMES)"
	@echo "  Example:   make replay FILE=datasets/recording_cent.touch SPEED=0.5"

# ---------------------------------------------------------------- setup
setup: ## Create .venv with uv and install requirements
	uv venv $(VENV)
	uv pip install --python $(VENV)/bin/python -r requirements.txt

# ---------------------------------------------------------------- task 1: recorded data
info: ## Print frames / duration / fps / LED config of every .touch file
	$(PY) touch_loader.py $(DATA)/*.touch

replay: ## Replay FILE in an OpenCV window (space pause, a/d step, s save, q quit)
	$(PY) replay.py $(FILE) --speed $(SPEED)

export: ## Export FILE as an mp4 to outputs/replay/ (no display needed)
	$(PY) replay.py $(FILE) --speed $(SPEED) --export

export-all: ## Export every .touch file as mp4
	@for f in $(DATA)/*.touch; do $(PY) replay.py $$f --export; done

frames: ## Save the TOP strongest-contact frames per object to outputs/frames/
	$(PY) save_interesting.py --data $(DATA) --top $(TOP)

# ---------------------------------------------------------------- task 2: illumination
illumination: ## Analyse recording_test.touch vs contactless recording.touch -> outputs/illumination/
	$(PY) analyze_illumination.py --test $(DATA)/recording_test.touch --ref $(DATA)/recording.touch

collect: ## Live DIGIT capture, global LED sweep 0..15 (free + contact per level)
	$(PY) collect_digit.py --serial $(SERIAL) --mode global --frames $(FRAMES)

collect-rgb: ## Live DIGIT capture, one LED colour at a time
	$(PY) collect_digit.py --serial $(SERIAL) --mode rgb --frames $(FRAMES)

collect-free: ## Live DIGIT capture, global sweep without the contact phase
	$(PY) collect_digit.py --serial $(SERIAL) --mode global --frames $(FRAMES) --no-contact

collect-sim: ## Simulated sweep (no sensor) - only for testing the pipeline
	$(PY) collect_digit.py --simulate --frames $(FRAMES)

analyze-sweep: ## Analyse the latest (or SWEEP=...) captured sweep
	@test -n "$(SWEEP)" || (echo "No sweep found - run 'make collect' first"; exit 1)
	$(PY) analyze_illumination.py --sweep $(SWEEP)

# ---------------------------------------------------------------- combos
all: info frames illumination export ## Run everything that works without the sensor

clean: ## Remove generated PNG/mp4 outputs (keeps collected sweeps)
	rm -rf outputs/frames outputs/illumination outputs/replay

clean-all: clean ## Also remove collected sweeps and the venv
	rm -rf outputs/collected $(VENV)

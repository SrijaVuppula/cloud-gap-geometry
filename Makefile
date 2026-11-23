# Reproduce the paper.
#   make paper      tables, figures and every in-text number from results/ (CPU, ~2 min)
#   make test       check them against the values printed in the paper
# GPU pipeline (writes to runs/, never overwrites the released results/):
#   make download splits eval-zs finetune distance energy

PY ?= python
BACKBONES ?= tiny 100M 300M 600M

.PHONY: paper stats tables numbers figures test download splits eval-zs finetune distance energy complexity clean

paper: stats tables figures numbers

stats:
	cd analysis && $(PY) compute_stats.py

tables: stats
	cd analysis && $(PY) tables.py

numbers: stats
	cd analysis && $(PY) paper_numbers.py

figures: stats
	cd analysis && $(PY) fig1_overview.py && $(PY) fig2_distance.py && $(PY) fig3_crossover.py

test:
	$(PY) -m pytest -q

download:
	$(PY) scripts/00_download.py

splits:
	$(PY) scripts/01_make_splits.py

eval-zs:
	for bb in $(BACKBONES); do $(PY) scripts/02_evaluate.py --backbone $$bb; done

finetune:
	BACKBONES="$(BACKBONES)" bash scripts/run_all_finetuning.sh

distance:
	for bb in $(BACKBONES); do $(PY) scripts/05_patch_distance.py --backbone $$bb; done

energy:
	$(PY) scripts/04_measure_energy.py
	$(PY) scripts/04_measure_energy.py --order reversed
	$(PY) scripts/04_measure_energy.py --finetuned --seed 1

complexity:
	$(PY) scripts/06_chip_complexity.py

clean:
	rm -rf results/derived tables figures/*.pdf

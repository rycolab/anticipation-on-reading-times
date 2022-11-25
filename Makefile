LANGUAGE := english
DATASET := natural_stories
MODEL := gpt-small
DATA_DIR := ./corpora
CHECKPOINT_DIR := ./checkpoints
RESULTS_DIR := ./results

DATASET_BASE_NAME := $(if $(filter-out $(DATASET), provo_skip2zero),$(DATASET),provo)
DATASET_BASE_NAME := $(if $(filter-out $(DATASET), dundee_skip2zero),$(DATASET_BASE_NAME),dundee)

SKIP_IN_DATASET := $(if $(filter-out $(DATASET), provo_skip2zero),False,True)
SKIP_IN_DATASET := $(if $(filter-out $(DATASET), dundee_skip2zero),$(SKIP_IN_DATASET),True)


COLA_URL := https://nyu-mll.github.io/CoLA/cola_public_1.1.zip
PROVO_URL1 := https://osf.io/a32be/download
PROVO_URL2 := https://osf.io/e4a2m/download
UCL_URL := https://static-content.springer.com/esm/art%3A10.3758%2Fs13428-012-0313-y/MediaObjects/13428_2012_313_MOESM1_ESM.zip
BNC_URL := https://gu-clasp.github.io/914a288ca1e127a7f1547412d9a7e056/bnc.csv

NS_URL1 := https://raw.githubusercontent.com/languageMIT/naturalstories/master/probs/all_stories_gpt3.csv
NS_URL2 := https://raw.githubusercontent.com/languageMIT/naturalstories/master/naturalstories_RTS/processed_RTs.tsv

NS_DIR := $(DATA_DIR)/natural_stories/
NS_FILE1 := $(NS_DIR)/all_stories_gpt3.csv
NS_FILE2 := $(NS_DIR)/processed_RTs.tsv

COLA_DIR_BASE := $(DATA_DIR)/cola/
COLA_FILE_RAW := $(COLA_DIR_BASE)/cola.zip
COLA_DIR := $(COLA_DIR_BASE)/cola_public/

PROVO_DIR := $(DATA_DIR)/provo/
PROVO_FILE1 := $(PROVO_DIR)/provo.csv
PROVO_FILE2 := $(PROVO_DIR)/provo_norms.csv

UCL_DIR := $(DATA_DIR)/ucl/
UCL_FILE_RAW := $(UCL_DIR)/ucl.zip
UCL_FILE := $(UCL_DIR)/stimuli_pos.txt

BROWN_DIR := $(DATA_DIR)/brown/
BROWN_FILE_RAW := $(BROWN_DIR)/data.zip
BROWN_FILE := $(BROWN_DIR)/brown_spr.csv

BNC_DIR := $(DATA_DIR)/bnc/
BNC_FILE := $(BROWN_DIR)/bnc.csv

DUNDEE_DIR := $(DATA_DIR)/dundee/
DUNDEE_FILE_RAW := $(DATA_DIR)/dundee.zip
DUNDEE_FILE := $(DUNDEE_DIR)/eye-tracking/sa01ma1p.dat


DATASET_DIR := $(DATA_DIR)/$(DATASET_BASE_NAME)
RT_ENTROPY_DIR := $(CHECKPOINT_DIR)/rt_and_entropy/
PARAMS_DIR := $(CHECKPOINT_DIR)/params/
DELTA_LLH_DIR := $(CHECKPOINT_DIR)/delta_llh/

PROCESSED_FILE := $(RT_ENTROPY_DIR)/rt_vs_entropy-$(DATASET)-$(MODEL).tsv

ANALYSIS_PARAMS_FULL_FNAME_BASE := $(PARAMS_DIR)/full-$(DATASET)-$(MODEL)
ANALYSIS_PARAMS_MERGED_FNAME_BASE := $(PARAMS_DIR)/merged-$(DATASET)-$(MODEL)
ANALYSIS_PARAMS_LINEAR_FNAME_BASE := $(PARAMS_DIR)/merged-linear-$(DATASET)-$(MODEL)
ANALYSIS_PARAMS_LINEAR_RENYI_FNAME_BASE := $(PARAMS_DIR)/merged-linear-renyi-$(DATASET)-$(MODEL)
ANALYSIS_PARAMS_LINEAR_SKIP_FNAME_BASE := $(PARAMS_DIR)/merged-linear-skip-$(DATASET)-$(MODEL)

LLH_LINEAR_FILE := $(DELTA_LLH_DIR)/merged-linear-$(DATASET)-$(MODEL).tsv
LLH_SKIP_FILE :=  $(DELTA_LLH_DIR)/skip-merged-linear-$(DATASET)-$(MODEL).tsv

ALL_DATASETS :=  brown natural_stories \
	provo provo_skip2zero provo4 provo4_skip2zero provo5 provo5_skip2zero \
	dundee dundee_skip2zero dundee4 dundee4_skip2zero dundee5 dundee5_skip2zero


ifneq ($(filter $(SKIP_IN_DATASET),True),)
all: get_data process_data get_llh_linear get_llh_skip
else
all: get_data process_data get_llh_linear
endif

get_llh_skip: $(LLH_SKIP_FILE)

get_llh_linear: $(LLH_LINEAR_FILE)

process_data: $(PROCESSED_FILE)

get_data: $(COLA_DIR) $(PROVO_FILE2) $(UCL_FILE) $(NS_FILE2) $(BROWN_FILE) $(BNC_FILE) $(DUNDEE_FILE)

plot_results:
	mkdir -p results/plots
	python src/h03_paper/plot_effects.py
	python src/h03_paper/plot_entropy_vs_surprisal.py
	python src/h03_paper/plot_renyi_llh.py
	python src/other/renyi_analysis_script.py

print_table_1:
	python src/h03_paper/print_table_1_surprisal.py

print_table_2:
	python src/h03_paper/print_table_2_entropy.py

print_table_3:
	python src/h03_paper/print_table_2_entropy.py --predictor renyi_0.50

print_table_4:
	python src/h03_paper/print_table_4_skip.py
	python src/h03_paper/print_table_4_skip.py --predictor renyi_0.50

print_table_5:
	python src/h03_paper/print_table_5_discard_skipped.py
	python src/h03_paper/print_table_5_discard_skipped.py --predictor renyi_0.50

print_table_6:
	python src/h03_paper/print_table_6_budgeting.py
	python src/h03_paper/print_table_6_budgeting.py --predictor renyi_0.50

print_table_7:
	python src/h03_paper/print_table_7_successorentropy.py
	python src/h03_paper/print_table_7_successorentropy.py --predictor renyi_0.50

print_table_8:
	Rscript src/h03_paper/print_table_app1_datasetstatistics.R 


# Get log-likelihoods
$(LLH_SKIP_FILE):
	mkdir -p $(PARAMS_DIR)
	mkdir -p $(DELTA_LLH_DIR)
	Rscript src/h02_rt_model/skip_vs_entropy.R $(PROCESSED_FILE) $(LLH_SKIP_FILE) $(ANALYSIS_PARAMS_LINEAR_SKIP_FNAME_BASE) --merge-workers --is-linear

$(LLH_LINEAR_FILE):
	mkdir -p $(PARAMS_DIR)
	mkdir -p $(DELTA_LLH_DIR)
	Rscript src/h02_rt_model/rt_vs_entropy.R $(PROCESSED_FILE) $(LLH_LINEAR_FILE) $(ANALYSIS_PARAMS_LINEAR_FNAME_BASE) --merge-workers --is-linear

# Preprocess rt data
$(PROCESSED_FILE):
	echo "Process rt data in " $(DATASET)
	mkdir -p $(RT_ENTROPY_DIR)
	python src/h01_data/get_surprisals.py --model $(MODEL) --dataset $(DATASET) --input-path $(DATASET_DIR) --output-fname $(PROCESSED_FILE)

# Get natural stories data
$(NS_FILE2):
	mkdir -p $(NS_DIR)
	wget -O $(NS_FILE1) $(NS_URL1)
	wget -O $(NS_FILE2) $(NS_URL2)

# Get BNC data
$(BNC_FILE):
	mkdir -p $(BNC_DIR)
	wget -O $(BNC_FILE) $(BNC_URL)

# Get brown data
$(BROWN_FILE):
	mkdir -p $(BROWN_DIR)
	gdown -O $(BROWN_DIR) https://drive.google.com/u/0/uc?id=1e-anJ4laGlTY-E0LNook1EzKBU2S1jI8
	unzip $(BROWN_FILE_RAW) -d $(BROWN_DIR)
	mv $(BROWN_DIR)/data/corpora/*brown* $(BROWN_DIR)/

# Get UCL data
$(UCL_FILE):
	mkdir -p $(UCL_DIR)
	wget -O $(UCL_FILE_RAW) $(UCL_URL)
	unzip $(UCL_FILE_RAW) -d $(UCL_DIR)

# Get PROVO data
$(PROVO_FILE2):
	mkdir -p $(PROVO_DIR)
	wget -O $(PROVO_FILE1) $(PROVO_URL1)
	wget -O $(PROVO_FILE2) $(PROVO_URL2)

# Get dundee data
$(DUNDEE_FILE):
	unzip $(DUNDEE_FILE_RAW) -d $(DATA_DIR)

# Get COLA data
$(COLA_DIR):
	mkdir -p $(COLA_DIR_BASE)
	wget -O $(COLA_FILE_RAW) $(COLA_URL)
	unzip $(COLA_FILE_RAW) -d $(COLA_DIR_BASE)

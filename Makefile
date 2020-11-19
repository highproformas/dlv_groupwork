.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3

-include .env
#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROFILE = default
PROJECT_NAME = dlv_groupwork
PYTHON_INTERPRETER = python3

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

ifeq (,$(shell which nvidia-docker))
DOCKER=docker
PLATFORM=cpu
else
DOCKER=nvidia-docker
PLATFORM=gpu
endif
PROJECT_NAME=dlv
DOCKERFILE=Dockerfile.$(PLATFORM)
IMAGE_NAME=$(PROJECT_NAME)-image-$(PLATFORM)
CONTAINER_NAME=$(PROJECT_NAME)-container-$(PLATFORM)
PWD=`pwd`
export JUPYTER_HOST_PORT=8888
export JUPYTER_CONTAINER_PORT=8888
export TENSORBOARD_HOST_PORT=6006
export TENSORBOARD_CONTAINER_PORT=6006

define START_DOCKER_CONTAINER
if [ `$(DOCKER) inspect -f {{.State.Running}} $(CONTAINER_NAME)` = "false" ] ; then
        $(DOCKER) start $(CONTAINER_NAME)
fi
endef


#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Make Dataset
data: 
	$(PYTHON_INTERPRETER) src/data/make_dataset.py ./data/raw ./data/processed

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8
lint:
	flake8 src


#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## init everything
init: clean-docker init-docker create-container start-container

## initialize docker image
init-docker: 
	$(DOCKER) build -t $(IMAGE_NAME) -f $(DOCKERFILE) .

## create docker container
create-container: 
	$(DOCKER) run -t -d $(FLAGS)-v $(PWD):/work -p $(JUPYTER_HOST_PORT):$(JUPYTER_CONTAINER_PORT) -p $(TENSORBOARD_HOST_PORT):$(TENSORBOARD_CONTAINER_PORT) --name $(CONTAINER_NAME) $(IMAGE_NAME)

## start docker container
start-container: 
	@echo "$$START_DOCKER_CONTAINER" | $(SHELL)
	@echo "Launched $(CONTAINER_NAME)..."
	$(DOCKER) attach $(CONTAINER_NAME)

## start Jupyter Lab server
jupyter: 
	jupyter-lab --allow-root --ip=0.0.0.0 --port=${JUPYTER_CONTAINER_PORT}

## fix versions
pip-freeze:
	pip freeze > ./requirements.txt

## remove Docker image and container
clean-docker: clean-container 

## there will be dragons
clean-docker-full-only-in-emergency: clean-container clean-image 

## remove Docker container
clean-container: 
	-$(DOCKER) rm $(CONTAINER_NAME)

## remove Docker image
clean-image: 
	-$(DOCKER) image rm $(IMAGE_NAME)

## download kaggle data
download-kaggle-data:
	kaggle datasets download -d mlg-ulb/creditcardfraud -p ./data/external --unzip
	cp ./data/external/* ./data/raw


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')

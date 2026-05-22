.PHONY: run validate serve all
 
all: run validate
 
run:
	python src/pipeline.py
 
validate:
	python validate.py

serve:
	uvicorn src.api:app --host 127.0.0.1 --port 8000

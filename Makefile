.PHONY: style check_code_quality

export PYTHONPATH = .
check_dirs := inference cli/inference_cli cli/inference_client

style:
	black  $(check_dirs)
	isort --profile black $(check_dirs)

check_code_quality:
	black --check $(check_dirs)
	# stop the build if there are Python syntax errors or undefined names
	flake8 $(check_dirs) --count --select=E9,F63,F7,F82 --show-source --statistics
	# exit-zero treats all errors as warnings. E203 for black, E501 for docstring, W503 for line breaks before logical operators 
	flake8 $(check_dirs) --count --max-line-length=88 --exit-zero  --ignore=D --extend-ignore=E203,E501,W503  --statistics

start_test_docker:
	docker run -d --rm -p $(PORT):$(PORT) -e PORT=$(PORT) --name inference-test roboflow/roboflow-inference-server-cpu:test

create_wheels:
	rm -f dist/*
	python .release/pypi/inference.core.setup.py bdist_wheel
	python .release/pypi/inference.cpu.setup.py bdist_wheel
	python .release/pypi/inference.gpu.setup.py bdist_wheel
	python .release/pypi/inference.setup.py bdist_wheel
	cd cli && python setup.py bdist_wheel & cd ..

upload_wheels:
	twine upload dist/*.whl
	twine upload cli/dist/*.whl

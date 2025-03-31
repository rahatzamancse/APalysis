.PHONY: all clean build_react build_python

# Define the build directories
REACT_DIR := frontend
PYTHON_DIR := backend
BUILD_DIR := $(PYTHON_DIR)/channelexplorer/static

all: clean build_react build_python

clean:
	@echo "Cleaning up previous builds..."
	rm -rf $(BUILD_DIR)
	rm -rf $(REACT_DIR)/build
	rm -rf $(PYTHON_DIR)/dist

build_react:
	@echo "Building React project..."
	mkdir -p $(BUILD_DIR)
	pnpm install --prefix $(REACT_DIR) && BUILD_PATH=../$(BUILD_DIR) pnpm run build --prefix $(REACT_DIR)

build_python:
	@echo "Building Python project with Poetry..."
	cd $(PYTHON_DIR) && poetry build
	
publish_python: clean build_react build_python
	@echo "Publishing Python project to PyPI..."
	cd $(PYTHON_DIR) && poetry publish --build

# Logic Synthesis Tool

Reads AIGER circuit files and converts them to MIG (Majority-Inverter Graph) networks using the mockturtle library.

## Fresh Clone Setup (Run On Another Machine)

Use these steps after cloning so training/inference scripts run without missing dependencies.

### 1. System Dependencies (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install -y build-essential cmake python3 python3-venv python3-pip libfmt-dev
```

### 2. Python Environment

From repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Build Native Binaries Required By 2_process

```bash
# Build import tool (1_inputs)
cmake -S 1_inputs -B 1_inputs/build
cmake --build 1_inputs/build -j

# Build feature extractor + MIG interface (2_process)
cmake -S 2_process -B 2_process/build
cmake --build 2_process/build -j

# Build ABC executable used by HybridSYN env
make -C 2_process/abc -j
```

Expected binaries:

```text
1_inputs/build/import
2_process/build/extract_features
2_process/build/interface
2_process/abc/abc
```

### 4. Quick Smoke Run

```bash
source .venv/bin/activate
python 2_process/train_hybridsyn_ppo.py \
	--circuit-file ../1_inputs/EPFL_benchmarks/arithmetic/adder.aig \
	--sequence-steps 2 \
	--ppo-steps 2 \
	--total-actions 2 \
	--log-file-name smoke_test.log \
	--seed 7
```

If this finishes, clone setup is complete.

## Build

From the workspace root:

```bash
cmake -S 1_inputs -B 1_inputs/build
cmake --build 1_inputs/build -j
```

The executable will be generated at `1_inputs/build/import`.

## Dependencies

- **C++17 compiler** (g++, clang, or equivalent)
- **CMake** >= 3.10
- **(Optional) libfmt-dev** — for system-wide fmt library; otherwise uses bundled copy

### Optional: Install system packages (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install build-essential cmake libfmt-dev
```

## Run

```bash
./1_inputs/build/import <circuit.aig>
```

### Example

```bash
./1_inputs/build/import 1_inputs/EPFL_benchmarks/arithmetic/adder.aig
```

Output:
```
Successfully read 1_inputs/EPFL_benchmarks/arithmetic/adder.aig and translated to MIG in RAM
Number of majority gates: <N>
```

## Project Structure

```
HybridSYN/
├── 1_inputs/                          # Input files and source code
│   ├── CMakeLists.txt                 # Build configuration
│   ├── import.cpp                     # Main program
│   ├── mockturtle/                    # Logic synthesis library
│   │   ├── include/                   # Public headers
│   │   └── lib/                       # Third-party dependencies
│   │       ├── lorina/                # AIGER/Verilog parser
│   │       ├── fmt/                   # String formatting
│   │       ├── kitty/                 # Truth table tools
│   │       └── parallel_hashmap/      # Hash map library
│   ├── EPFL_benchmarks/               # Test circuits
│   └── build/                         # Build artifacts (after cmake)
├── 2_process/                         # Processing/training code
├── 3_outputs/                         # Output results
└── README.md                          # This file
```

## Key Include Paths

The CMakeLists.txt automatically configures include paths for:
- `<mockturtle/...>` — main library headers
- `<lorina/...>` — AIGER/file parsers
- `<fmt/...>` — formatting library
- `<kitty/...>` — truth table utilities
- `<parallel_hashmap/...>` — concurrent hash maps

## Troubleshooting

**"No such file or directory" errors during compile:**
- Ensure you're running `cmake` from the workspace root
- Verify all include directories exist in `mockturtle/lib/`
- Check that C++17 is available: `g++ --version` or `clang --version`

**Symbol errors (e.g., "lorina::aiger_reader not found"):**
- Ensure the correct namespaces are used: `lorina::read_aiger`, `lorina::return_code`, `mockturtle::aiger_reader`
- Verify headers are included: `<lorina/aiger.hpp>`, `<mockturtle/io/aiger_reader.hpp>`, `<mockturtle/networks/mig.hpp>`

## License

Mockturtle and dependencies are provided under their respective licenses (see `mockturtle/LICENSE` and component-specific license files in `mockturtle/lib/`).

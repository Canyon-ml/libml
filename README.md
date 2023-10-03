
# LibML

LibML is a thorougly documented, rigorously tested, implementation of ONNX operators for machine learning and
their derivatives.  LibML makes it easy to implement highly performant, correct Machine Learning APIs.  LibML doesn't care how you implement your API, it just provides the math. 

Note: LibML is currently in very early development and should not be used in production.

## Goals

- [ ] Implement every ONNX operator
- [ ] Support CPU and GPU 
- [ ] Provide formal proofs and definitions for each operator
- [ ] Write benchmarks for each operator

## Contribution

If you need an operator and find it hasn't been implemented yet, feel free to create a pull request.  We also need contributors to help peer-review and write formal proofs in LaTeX for the operators and write benchmarks. 

## Installation

LibML is a native rust api, so install with `cargo add libml`.

## Usage

All forward operators in libml have the form:
```
libml::<op-name>(<args>);
```

Backward operators have the form:
```
libml::<op-name>_wrt_<arg>(<args>);
```

## Documentation

All the operators have module-level and function documentation on docs.rs. For more in-depth docs, see `operators.pdf` in /docs.




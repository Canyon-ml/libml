
# LibML

Note: LibML is currently in very early development and is subject to frequent, breaking changes. 

LibML is a thorougly documented, rigorously tested, high-performance implementation of ONNX operators for machine learning and
their derivatives. As the quantity and complexity of operators continues to grow, we need to create ways to manage that complexity. Libml solves this problem by being _very_ explicit about the inner workings of its functions, and following _consistent_ design patterns so you can be certain your code will behave the way you want it to.  Rust is the perfect candidate for such a library, with its strong static typing, borrow / mutability restraints, and documentation features. 

## Goals

- [ ] Implement major ONNX Operators
- [ ] Support CPU and GPU 
- [ ] Provide formal proofs and definitions for operators.
- [ ] Write benchmarks for each operator and maximize performance

## Short-term goals

Make sure every operator is _working_ before optimizing for performance. 

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




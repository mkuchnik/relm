use pyo3::prelude::*;
use rustfst::prelude::*;
use regex_syntax::Parser;
use rust_regex_compiler::RegexToFSTBuilder;

/// Takes a regex and returns the fst representation.
#[pyfunction]
fn regex_to_fst(regex: String) -> PyResult<String> {
    let fst = RegexToFSTBuilder::new().parse(&regex).unwrap();
    let fst_str = fst.text().unwrap().to_string();
    Ok(fst_str)
}

/// Takes a regex and returns HIR representation.
#[pyfunction]
fn regex_to_hir(regex: String) -> PyResult<String> {
    let hir = Parser::new().parse(&regex).unwrap();
    let hir_string: String = format!("{}", hir);
    Ok(hir_string)
}

/// A Python module implemented in Rust.
#[pymodule]
fn rust_regex_compiler_bindings(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(regex_to_fst, m)?)?;
    m.add_function(wrap_pyfunction!(regex_to_hir, m)?)?;
    Ok(())
}
/*
    Copyright (C) 2023 Michael Kuchnik. All Right Reserved.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

use std::fmt;
use regex_syntax::hir;
use regex_syntax::hir::{Hir, HirKind};
use regex_syntax::Parser;
use regex_syntax::hir::Visitor;
use rustfst::prelude::*;
use rustfst::algorithms::determinize::determinize;
use rustfst::algorithms::rm_epsilon::rm_epsilon;
use pyo3::prelude::*;

type FSTContainerType = TropicalWeight;
type FSTContainer = VectorFst::<FSTContainerType>;

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
#[pyo3(name="relm_rust_regex_compiler_bindings")]
fn relm_rust_regex_compiler_bindings(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(regex_to_fst, m)?)?;
    m.add_function(wrap_pyfunction!(regex_to_hir, m)?)?;
    Ok(())
}

pub struct RegexToFSTBuilder {}

impl RegexToFSTBuilder {
    pub fn new() -> Self {
        Self{}
    }

    pub fn parse(&self, regex: &str) -> Result<FSTContainer, Box<dyn std::error::Error>> {
        let hir = Parser::new().parse(&regex).unwrap();
        let mut my_stack_builder = FSTStack::new();
        let visitor = ReASTVisitor::new(&mut my_stack_builder);
        regex_syntax::hir::visit(&hir, visitor).unwrap();
        if my_stack_builder.stack.len() != 1 {
            panic!("Expected result stack to contain 1 element. Found {} ({:#?})",
                   my_stack_builder.stack.len(), my_stack_builder.stack);
        }
        let mut fst = my_stack_builder.stack.pop().unwrap();
        connect(&mut fst)?;
        rm_epsilon(&mut fst)?;
        fst = determinize(&fst)?;
        minimize(&mut fst)?;
        Ok(fst)
    }
}

#[derive(Debug)]
struct FSTStack {
    stack: Vec<FSTContainer>,
}

impl FSTStack {
    fn new() -> Self {
        Self {
            stack: vec![],
        }
    }
}

#[derive(Debug)]
struct ReASTVisitor<'r> {
    fst_builder: &'r mut FSTStack,
}

impl<'r> Visitor for ReASTVisitor<'r> {
    // See Writer impl in https://github.com/rust-lang/regex/blob/master/regex-syntax/src/hir/print.rs
    // See also Regex src/compile.rs for a code-generation method
    type Output = ();
    type Err = fmt::Error;

    fn finish(self) -> fmt::Result {
        Ok(())
    }

    fn visit_pre(&mut self, _hir: &Hir) -> fmt::Result {
        Ok(())
    }

    fn visit_post(&mut self, hir: &Hir) -> fmt::Result {
        match *hir.kind() {
            HirKind::Empty => {
                // NOTE(mkuchnik): We assume "" matches only on ""
                let fst = FSTEmptyAcceptor::new().emit().unwrap();
                self.fst_builder.stack.push(fst);
            }
            HirKind::Literal(hir::Literal::Unicode(c)) => {
                let fst = FSTLiteral::new(c as u32).emit().unwrap();
                self.fst_builder.stack.push(fst);
            }
            HirKind::Literal(hir::Literal::Byte(b)) => {
                let fst = FSTLiteral::new(b as u32).emit().unwrap();
                self.fst_builder.stack.push(fst);
            }
            HirKind::Class(hir::Class::Unicode(ref cls)) => {
                let mut internal_stack: Vec<FSTContainer> = vec![];
                for range in cls.iter() {
                    if range.start() == range.end() {
                        let fst = FSTLiteral::new(range.start() as u32).emit().unwrap();
                        internal_stack.push(fst);
                    } else {
                        let literal_range: Vec<u32> = (range.start()..=range.end())
                            .map(|c| c as u32)
                            .collect();
                        let fst = FSTRangeLiteral::new(literal_range).emit().unwrap();
                        internal_stack.push(fst);
                    }
                }
                match internal_stack.len() {
                    0 => {
                        panic!("Found 0 elements emitted");
                    },
                    1 => {
                        let e = internal_stack.pop().expect("Pop should always succeed");
                        self.fst_builder.stack.push(e);
                    },
                    _ => {
                        let fst = FSTAlteration::new(internal_stack).emit().unwrap();
                        self.fst_builder.stack.push(fst);
                    },
                }
            }
            HirKind::Class(hir::Class::Bytes(ref cls)) => {
                let mut internal_stack: Vec<FSTContainer> = vec![];
                for range in cls.iter() {
                    if range.start() == range.end() {
                        let fst = FSTLiteral::new(range.start() as u32).emit().unwrap();
                        internal_stack.push(fst);
                    } else {
                        let literal_range: Vec<u32> = (range.start()..=range.end())
                            .map(|c| c as u32)
                            .collect();
                        let fst = FSTRangeLiteral::new(literal_range).emit().unwrap();
                        internal_stack.push(fst);
                    }
                }
                match internal_stack.len() {
                    0 => {
                        panic!("Found 0 elements emitted");
                    },
                    1 => {
                        let e = internal_stack.pop().expect("Pop should always succeed");
                        self.fst_builder.stack.push(e);
                    },
                    _ => {
                        let fst = FSTAlteration::new(internal_stack).emit().unwrap();
                        self.fst_builder.stack.push(fst);
                    },
                }
            }
            HirKind::Anchor(ref x) => {
                panic!("No Emit Anchor {:#?}", x);
            }
            HirKind::WordBoundary(ref x) => {
                panic!("No Emit WordBoundary {:#?}", x);
            }
            HirKind::Concat(ref x) => {
                assert!(x.len() > 0);
                let mut fsts = vec![FSTContainer::new(); x.len()];
                for i in (0..x.len()).rev() {
                    fsts[i] = self.fst_builder.stack.pop().unwrap();
                }
                let fst = FSTConcatenation::new(fsts).emit().unwrap();
                self.fst_builder.stack.push(fst);
            }
            HirKind::Alternation(ref x) => {
                assert!(x.len() > 0);
                let mut fsts = vec![FSTContainer::new(); x.len()];
                for i in (0..x.len()).rev() {
                    fsts[i] = self.fst_builder.stack.pop().unwrap();
                }
                let fst = FSTAlteration::new(fsts).emit().unwrap();
                self.fst_builder.stack.push(fst);
            }
            HirKind::Repetition(ref x) => {
                match x.kind {
                    hir::RepetitionKind::ZeroOrOne => {
                        let fst = self.fst_builder.stack.pop().unwrap();
                        let fst = FSTFiniteRepetition::new(fst, 0, 1).emit().unwrap();
                        self.fst_builder.stack.push(fst);
                    }
                    hir::RepetitionKind::ZeroOrMore => {
                        let fst = self.fst_builder.stack.pop().unwrap();
                        let fst = FSTStarRepetition::new(fst).emit().unwrap();
                        self.fst_builder.stack.push(fst);
                    }
                    hir::RepetitionKind::OneOrMore => {
                        let fst = self.fst_builder.stack.pop().unwrap();
                        let fst = FSTPlusRepetition::new(fst).emit().unwrap();
                        self.fst_builder.stack.push(fst);
                    }
                    hir::RepetitionKind::Range(ref x) => match *x {
                        hir::RepetitionRange::Exactly(m) => {
                            let fst = self.fst_builder.stack.pop().unwrap();
                            let fst = FSTFiniteRepetition::new(fst, m, m).emit().unwrap();
                            self.fst_builder.stack.push(fst);
                        }
                        hir::RepetitionRange::AtLeast(m) => {
                            let fst = self.fst_builder.stack.pop().unwrap();
                            let fst_exact = FSTFiniteRepetition::new(fst.clone(), m, m).emit().unwrap();
                            let fst_star = FSTStarRepetition::new(fst).emit().unwrap();
                            let fsts = vec![fst_exact, fst_star];
                            let fst = FSTConcatenation::new(fsts).emit().unwrap();
                            self.fst_builder.stack.push(fst);
                        }
                        hir::RepetitionRange::Bounded(m, n) => {
                            let fst = self.fst_builder.stack.pop().unwrap();
                            let fst = FSTFiniteRepetition::new(fst, m, n).emit().unwrap();
                            self.fst_builder.stack.push(fst);
                        }
                    },
                }
                if !x.greedy {
                    panic!("No Emit Repetition {:#?}", x);
                }
            }
            HirKind::Group(_) => {
                // Do nothing
            }
        }
        Ok(())
    }
}

trait FSTEmittable {
    fn emit(&self) -> Result<FSTContainer, Box<dyn std::error::Error>>;
}

struct FSTEmptyAcceptor {}

impl FSTEmptyAcceptor {
    fn new() -> Self {
        Self{}
    }
}

impl FSTEmittable for FSTEmptyAcceptor {
    fn emit(&self) -> Result<FSTContainer, Box<dyn std::error::Error>> {
        let zero = FSTContainerType::new(0.);
        let mut fst = FSTContainer::new();
        let s0 = fst.add_state();
        fst.set_start(s0)?;
        fst.set_final(s0, zero)?;
        connect(&mut fst)?;
        Ok(fst)
    }
}

struct FSTLiteral {
    c: u32
}

impl FSTLiteral {
    fn new(c: u32) -> Self {
        Self{ c }
    }
}

impl FSTEmittable for FSTLiteral {
    fn emit(&self) -> Result<FSTContainer, Box<dyn std::error::Error>> {
        let t = self.c;
        let zero = FSTContainerType::new(0.);
        let mut fst = FSTContainer::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        // ilable, olabel, weight, nextstate
        // NOTE(mkuchnik): We just make every output zero
        let tr_1 = Tr::<FSTContainerType>::new(t, t, zero, s1);
        fst.add_tr(s0, tr_1)?;
        fst.set_start(s0)?;
        fst.set_final(s1, zero)?;
        connect(&mut fst)?;
        Ok(fst)
    }
}

struct FSTRangeLiteral {
    cs: Vec<u32>
}

impl FSTRangeLiteral {
    fn new(cs: Vec<u32>) -> Self {
        Self{ cs }
    }
}

impl FSTEmittable for FSTRangeLiteral {
    fn emit(&self) -> Result<FSTContainer, Box<dyn std::error::Error>> {
        let zero = FSTContainerType::new(0.);
        let mut fst = FSTContainer::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        // ilable, olabel, weight, nextstate
        // NOTE(mkuchnik): We just make every output zero
        for t in self.cs.iter() {
            let tr_1 = Tr::<FSTContainerType>::new(*t, *t, zero, s1);
            fst.add_tr(s0, tr_1)?;
        }
        fst.set_start(s0)?;
        fst.set_final(s1, zero)?;
        connect(&mut fst)?;
        Ok(fst)
    }
}

struct FSTConcatenation {
    fsts: Vec<FSTContainer>,
}

impl FSTConcatenation {
    fn new(fsts: Vec<FSTContainer>) -> Self {
        FSTConcatenation{ fsts }
    }
}

impl FSTEmittable for FSTConcatenation {
    fn emit(&self) -> Result<FSTContainer, Box<dyn std::error::Error>> {
        if self.fsts.len() < 2 {
            panic!("Not enough values to concatenate. Found {}.",
                   self.fsts.len());
        }
        let mut it = self.fsts.iter();
        let fst = it.next().unwrap().clone();
        let fst = it.fold(fst, |mut a, b| {
                concat::concat(&mut a, b).unwrap();
                a
            });
        Ok(fst)
    }
}

struct FSTStarRepetition {
    fst: FSTContainer,
}

impl FSTStarRepetition {
    fn new(fst: FSTContainer) -> Self {
        FSTStarRepetition{ fst }
    }
}

impl FSTEmittable for FSTStarRepetition {
    fn emit(&self) -> Result<FSTContainer, Box<dyn std::error::Error>> {
        let mut fst = self.fst.clone();  // TODO(mkuchnik): Copy
        closure::closure(&mut fst, closure::ClosureType::ClosureStar);
        Ok(fst)
    }
}

struct FSTPlusRepetition {
    fst: FSTContainer,
}

impl FSTPlusRepetition {
    fn new(fst: FSTContainer) -> Self {
        FSTPlusRepetition{ fst }
    }
}

impl FSTEmittable for FSTPlusRepetition {
    fn emit(&self) -> Result<FSTContainer, Box<dyn std::error::Error>> {
        let mut fst = self.fst.clone();  // TODO(mkuchnik): Copy
        closure::closure(&mut fst, closure::ClosureType::ClosurePlus);
        Ok(fst)
    }
}

struct FSTFiniteRepetition {
    fst: FSTContainer,
    min: u32,
    max: u32,
}

impl FSTFiniteRepetition {
    fn new(fst: FSTContainer, min: u32, max: u32) -> Self {
        if max < min {
            panic!("Max {} < Min {}", max, min);
        }
        FSTFiniteRepetition{ fst, min, max }
    }
    fn repeat(fst: FSTContainer, amount: u32) -> FSTContainer {
        let amount: usize = usize::try_from(amount).unwrap();
        match amount {
            0 => {
                FSTEmptyAcceptor::new().emit().unwrap()
            }
            1 => {
                fst
            }
            _ => {
                let fsts = vec![fst; amount];
                let fst = FSTConcatenation::new(fsts).emit().unwrap();
                fst
            }
        }
    }
}

impl FSTEmittable for FSTFiniteRepetition {
    fn emit(&self) -> Result<FSTContainer, Box<dyn std::error::Error>> {
        // TODO(mkuchnik): This is at least quadratic because of repeated clones
        let mut last_fst: FSTContainer = FSTFiniteRepetition::repeat(self.fst.clone(), self.min);
        let mut fsts: Vec<FSTContainer> = vec![last_fst];
        for _ in (self.min+1)..=self.max {
            last_fst = fsts.last().unwrap().clone();
            let cat_fsts = vec![last_fst, self.fst.clone()];
            let new_fst = FSTConcatenation::new(cat_fsts).emit().unwrap();
            fsts.push(new_fst);
        }
        match fsts.len() {
            0 => panic!("Fsts are of length 0"),
            1 => Ok(fsts[0].clone()),
            _ => {
                let fst = FSTAlteration::new(fsts).emit().unwrap();
                Ok(fst)
            }
        }
    }
}

struct FSTAlteration {
    fsts: Vec<FSTContainer>,
}

impl FSTAlteration {
    fn new(fsts: Vec<FSTContainer>) -> Self {
        FSTAlteration{ fsts }
    }
}

impl FSTEmittable for FSTAlteration {
    fn emit(&self) -> Result<FSTContainer, Box<dyn std::error::Error>> {
        if self.fsts.len() < 2 {
            panic!("Not enough values to union. Found {}.",
                   self.fsts.len());
        }
        let mut it = self.fsts.iter();
        let fst = it.next().unwrap().clone();
        let fst = it.fold(fst, |mut a, b| {
                union::union(&mut a, b).unwrap();
                a
            });
        Ok(fst)
    }
}

impl<'r> ReASTVisitor<'r> {
    fn new(fst_builder: &'r mut FSTStack) -> Self {
        Self {
            fst_builder,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn test_literal() {
        let regex = "a";
        let fst = RegexToFSTBuilder::new().parse(&regex).unwrap();
        let paths : HashSet<_> = fst.paths_iter().collect();
        let mut paths_ref = HashSet::<FstPath<FSTContainerType>>::new();
        paths_ref.insert(fst_path![97 => 97]);
        assert_eq!(paths, paths_ref);
    }

    #[test]
    fn test_concatenation() {
        let regex = "ab";
        let fst = RegexToFSTBuilder::new().parse(&regex).unwrap();
        let paths : HashSet<_> = fst.paths_iter().collect();
        let mut paths_ref = HashSet::<FstPath<FSTContainerType>>::new();
        paths_ref.insert(fst_path![97,98 => 97,98]);
        assert_eq!(paths, paths_ref);
    }

    #[test]
    fn test_alteration() {
        let regex = "a|b";
        let fst = RegexToFSTBuilder::new().parse(&regex).unwrap();
        let paths : HashSet<_> = fst.paths_iter().collect();
        let mut paths_ref = HashSet::<FstPath<FSTContainerType>>::new();
        paths_ref.insert(fst_path![97 => 97]);
        paths_ref.insert(fst_path![98 => 98]);
        assert_eq!(paths, paths_ref);
    }

    #[test]
    fn test_repetition_star() {
        let regex = "a*";
        let fst = RegexToFSTBuilder::new().parse(&regex).unwrap();
        let max_samples = 5;
        // Sample paths
        let paths : HashSet<_> = fst.paths_iter().take(max_samples).collect();
        let mut paths_ref = HashSet::<FstPath<FSTContainerType>>::new();
        paths_ref.insert(fst_path![]);
        paths_ref.insert(fst_path![97 => 97]);
        paths_ref.insert(fst_path![97,97 => 97,97]);
        paths_ref.insert(fst_path![97,97,97 => 97,97,97]);
        paths_ref.insert(fst_path![97,97,97,97 => 97,97,97,97]);
        assert_eq!(paths, paths_ref);
    }

    #[test]
    fn test_repetition_plus() {
        let regex = "a+";
        let fst = RegexToFSTBuilder::new().parse(&regex).unwrap();
        let max_samples = 5;
        // Sample paths
        let paths : HashSet<_> = fst.paths_iter().take(max_samples).collect();
        let mut paths_ref = HashSet::<FstPath<FSTContainerType>>::new();
        paths_ref.insert(fst_path![97 => 97]);
        paths_ref.insert(fst_path![97,97 => 97,97]);
        paths_ref.insert(fst_path![97,97,97 => 97,97,97]);
        paths_ref.insert(fst_path![97,97,97,97 => 97,97,97,97]);
        paths_ref.insert(fst_path![97,97,97,97,97 => 97,97,97,97,97]);
        assert_eq!(paths, paths_ref);
    }

    #[test]
    fn test_repetition_exact() {
        let regex = "a{2}";
        let fst = RegexToFSTBuilder::new().parse(&regex).unwrap();
        let paths : HashSet<_> = fst.paths_iter().collect();
        let mut paths_ref = HashSet::<FstPath<FSTContainerType>>::new();
        paths_ref.insert(fst_path![97,97 => 97,97]);
        assert_eq!(paths, paths_ref);
    }

    #[test]
    fn test_repetition_at_least() {
        let regex = "a{2,}";
        let fst = RegexToFSTBuilder::new().parse(&regex).unwrap();
        let max_samples = 5;
        // Sample paths
        let paths : HashSet<_> = fst.paths_iter().take(max_samples).collect();
        let mut paths_ref = HashSet::<FstPath<FSTContainerType>>::new();
        paths_ref.insert(fst_path![97,97 => 97,97]);
        paths_ref.insert(fst_path![97,97,97 => 97,97,97]);
        paths_ref.insert(fst_path![97,97,97,97 => 97,97,97,97]);
        paths_ref.insert(fst_path![97,97,97,97,97 => 97,97,97,97,97]);
        paths_ref.insert(fst_path![97,97,97,97,97,97 => 97,97,97,97,97,97]);
        assert_eq!(paths, paths_ref);
    }

    #[test]
    fn test_repetition_range() {
        let regex = "a{2,5}";
        let fst = RegexToFSTBuilder::new().parse(&regex).unwrap();
        let paths : HashSet<_> = fst.paths_iter().collect();
        let mut paths_ref = HashSet::<FstPath<FSTContainerType>>::new();
        paths_ref.insert(fst_path![97,97 => 97,97]);
        paths_ref.insert(fst_path![97,97,97 => 97,97,97]);
        paths_ref.insert(fst_path![97,97,97,97 => 97,97,97,97]);
        paths_ref.insert(fst_path![97,97,97,97,97 => 97,97,97,97,97]);
        assert_eq!(paths, paths_ref);
    }

    #[test]
    fn test_optional() {
        let regex = "a?";
        let fst = RegexToFSTBuilder::new().parse(&regex).unwrap();
        let paths : HashSet<_> = fst.paths_iter().collect();
        let mut paths_ref = HashSet::<FstPath<FSTContainerType>>::new();
        paths_ref.insert(fst_path![]);
        paths_ref.insert(fst_path![97 => 97]);
        assert_eq!(paths, paths_ref);
    }

    #[test]
    fn test_group_alphabetical() {
        let regex = "[a-e]";
        let fst = RegexToFSTBuilder::new().parse(&regex).unwrap();
        let paths : HashSet<_> = fst.paths_iter().collect();
        let mut paths_ref = HashSet::<FstPath<FSTContainerType>>::new();
        for l in 'a' as u32..=('e' as u32) {
            paths_ref.insert(fst_path![l => l]);
        }
        assert_eq!(paths, paths_ref);
    }

    #[test]
    fn test_group_numeric() {
        let regex = "[0-9]";
        let fst = RegexToFSTBuilder::new().parse(&regex).unwrap();
        let paths : HashSet<_> = fst.paths_iter().collect();
        let mut paths_ref = HashSet::<FstPath<FSTContainerType>>::new();
        for l in '0' as u32..=('9' as u32) {
            paths_ref.insert(fst_path![l => l]);
        }
        assert_eq!(paths, paths_ref);
    }

    #[test]
    fn test_combinations() {
        let regex = "(a)|(bc)|(d{1,2})";
        let fst = RegexToFSTBuilder::new().parse(&regex).unwrap();
        let paths : HashSet<_> = fst.paths_iter().collect();
        let mut paths_ref = HashSet::<FstPath<FSTContainerType>>::new();
        paths_ref.insert(fst_path![97 => 97]);
        paths_ref.insert(fst_path![98,99 => 98,99]);
        paths_ref.insert(fst_path![100 => 100]);
        paths_ref.insert(fst_path![100,100 => 100,100]);
        assert_eq!(paths, paths_ref);
    }

    #[test]
    fn test_small_range() {
        let regex = "(([a-z]){0,2})";
        let fst = RegexToFSTBuilder::new().parse(&regex).unwrap();
        let paths_count = fst.paths_iter().count();
        let expected_num_paths = 1 + 26 + usize::pow(26, 2);
        assert_eq!(paths_count, expected_num_paths);
    }

    #[test]
    fn test_mini_mixed_range() {
        let regex = "(([a-b]|[A-B]|[0-1]){0,2})";
        let fst = RegexToFSTBuilder::new().parse(&regex).unwrap();
        let paths_count = fst.paths_iter().count();
        let num_choices = 2 + 2 + 2;
        let expected_num_paths = 1 + num_choices + usize::pow(num_choices, 2);
        assert_eq!(paths_count, expected_num_paths);
    }

    #[test]
    fn test_small_mixed_range() {
        let regex = "(([a-z]|[A-Z]|[0-9]){0,2})";
        let fst = RegexToFSTBuilder::new().parse(&regex).unwrap();
        let paths_count = fst.paths_iter().count();
        let num_choices = 26 * 2 + 10;
        let expected_num_paths = 1 + num_choices + usize::pow(num_choices, 2);
        assert_eq!(paths_count, expected_num_paths);
    }

    #[test]
    fn test_big_range() {
        let regex = "A(([a-z]){0,2})[0-9]";
        let fst = RegexToFSTBuilder::new().parse(&regex).unwrap();
        let paths_count = fst.paths_iter().count();
        let expected_num_paths = 1 * (1 + (26) + usize::pow(26, 2)) * 10;
        assert_eq!(paths_count, expected_num_paths);
    }

    #[test]
    fn test_big_range_optional() {
        let regex = "(A(([a-z]|(_)|(-)){0,2})[0-9])?";
        let fst = RegexToFSTBuilder::new().parse(&regex).unwrap();
        let paths_count = fst.paths_iter().count();
        let expected_num_paths = 1 * (1 + (26+2) + usize::pow(26+2, 2)) * 10 + 1;
        assert_eq!(paths_count, expected_num_paths);
    }

    #[test]
    fn test_mini_huge() {
        let valid_char = "([a-z]|[A-Z]|[0-9]|(_)|(-))";
        let regex = format!("([a-z]|[A-Z]|[0-9])({valid_char}{{0,2}})", valid_char=valid_char)
            .to_string();
        let fst = RegexToFSTBuilder::new().parse(&regex).unwrap();
        let paths_count = fst.paths_iter().count();
        let expected_num_paths = 62 * (1 + 64 + usize::pow(64, 2));
        assert_eq!(paths_count, expected_num_paths);
    }

    #[test]
    fn test_empty() {
        let regex = "";
        let fst = RegexToFSTBuilder::new().parse(&regex).unwrap();
        let paths : HashSet<_> = fst.paths_iter().collect();
        let mut paths_ref = HashSet::<FstPath<FSTContainerType>>::new();
        // NOTE(mkuchnik): Assume empty string only matches
        paths_ref.insert(fst_path![]);
        assert_eq!(paths, paths_ref);
    }

    #[test]
    fn test_range_literals() {
        let inputs: Vec<char> = ('a'..='z').collect();
        let fsts = inputs
            .iter()
            .map(|c| FSTLiteral::new(*c as u32).emit().unwrap())
            .collect();
        let fst = FSTAlteration::new(fsts).emit().unwrap();
        let u32_inputs: Vec<u32> = inputs.iter().map(|c| *c as u32).collect();
        let range_fst = FSTRangeLiteral::new(u32_inputs).emit().unwrap();
        let paths : HashSet<_> = fst.paths_iter().collect();
        let paths_ref : HashSet<_> = range_fst.paths_iter().collect();
        assert_eq!(paths, paths_ref);

    }

    #[test]
    fn test_dot() {
        let regex = ".";
        let fst = RegexToFSTBuilder::new().parse(&regex).unwrap();
        let paths_count = fst.paths_iter().count();
        let num_unicode = 1112063;  // Full set
        assert_eq!(paths_count, num_unicode);
    }

    #[test]
    fn test_redundant() {
        let regex = "a|a|a|a|a|a|(a|a)";
        let fst = RegexToFSTBuilder::new().parse(&regex).unwrap();
        let paths : HashSet<_> = fst.paths_iter().collect();
        let mut paths_ref = HashSet::<FstPath<FSTContainerType>>::new();
        paths_ref.insert(fst_path![97 => 97]);
        assert_eq!(paths, paths_ref);
    }

}

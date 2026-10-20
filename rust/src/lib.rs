use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::BTreeSet;

#[pyclass]
#[derive(Clone)]
pub struct RoutingCore {
    edges: Vec<(usize, usize)>,
    gates: Vec<Vec<usize>>,
    mapping: Vec<usize>,
    inverse: Vec<usize>,
    adjacent: Vec<Vec<bool>>,
    deps: Vec<usize>,
    children: Vec<Vec<usize>>,
    done: Vec<bool>,
    ready: BTreeSet<usize>,
}
#[pymethods]
impl RoutingCore {
    #[new]
    fn new(
        n: usize,
        edges: Vec<(usize, usize)>,
        gates: Vec<Vec<usize>>,
        mapping: Vec<usize>,
    ) -> PyResult<Self> {
        let mut sorted = mapping.clone();
        sorted.sort();
        if sorted != (0..n).collect::<Vec<_>>() {
            return Err(PyValueError::new_err(
                "mapping must be a complete permutation",
            ));
        }
        let mut adjacent = vec![vec![false; n]; n];
        for &(a, b) in &edges {
            if a >= n || b >= n || a == b {
                return Err(PyValueError::new_err("invalid edge"));
            }
            adjacent[a][b] = true;
            adjacent[b][a] = true;
        }
        let mut inverse = vec![0; n];
        for (q, &p) in mapping.iter().enumerate() {
            inverse[p] = q;
        }
        let mut last = vec![None; n];
        let mut deps = vec![0; gates.len()];
        let mut children = vec![vec![]; gates.len()];
        for (i, gate) in gates.iter().enumerate() {
            let unique: BTreeSet<usize> = gate.iter().copied().collect();
            if unique.len() != gate.len() || gate.iter().any(|&q| q >= n) {
                return Err(PyValueError::new_err("invalid gate qubits"));
            }
            let parents: BTreeSet<usize> = gate.iter().filter_map(|&q| last[q]).collect();
            deps[i] = parents.len();
            for p in parents {
                children[p].push(i);
            }
            for &q in gate {
                last[q] = Some(i);
            }
        }
        let ready = deps
            .iter()
            .enumerate()
            .filter_map(|(i, &d)| if d == 0 { Some(i) } else { None })
            .collect();
        let done = vec![false; gates.len()];
        Ok(Self {
            edges,
            gates,
            mapping,
            inverse,
            adjacent,
            deps,
            children,
            done,
            ready,
        })
    }
    fn mapping(&self) -> Vec<usize> {
        self.mapping.clone()
    }
    fn front(&self) -> Vec<usize> {
        self.ready.iter().copied().collect()
    }
    fn remaining(&self) -> Vec<usize> {
        self.done
            .iter()
            .enumerate()
            .filter_map(|(i, &d)| if !d { Some(i) } else { None })
            .collect()
    }
    fn drain(&mut self) -> Vec<usize> {
        let mut executed = Vec::new();
        loop {
            let ready: Vec<usize> = self
                .ready
                .iter()
                .copied()
                .filter(|&i| {
                    let g = &self.gates[i];
                    g.len() != 2 || self.adjacent[self.mapping[g[0]]][self.mapping[g[1]]]
                })
                .collect();
            if ready.is_empty() {
                break;
            }
            for i in ready {
                self.ready.remove(&i);
                self.done[i] = true;
                executed.push(i);
                for &c in &self.children[i] {
                    self.deps[c] -= 1;
                    if self.deps[c] == 0 {
                        self.ready.insert(c);
                    }
                }
            }
        }
        executed
    }
    fn mask(&self) -> Vec<bool> {
        let mut involved = vec![false; self.mapping.len()];
        for &i in &self.ready {
            for &q in &self.gates[i] {
                involved[self.mapping[q]] = true;
            }
        }
        self.edges
            .iter()
            .map(|&(a, b)| involved[a] || involved[b])
            .collect()
    }
    fn swap(&mut self, action: usize) -> PyResult<()> {
        if action >= self.edges.len() {
            return Err(PyValueError::new_err("invalid edge action"));
        }
        let (a, b) = self.edges[action];
        let (x, y) = (self.inverse[a], self.inverse[b]);
        self.mapping[x] = b;
        self.mapping[y] = a;
        self.inverse.swap(a, b);
        Ok(())
    }
    #[pyo3(name = "clone")]
    fn clone_py(&self) -> Self {
        Clone::clone(self)
    }
}
#[pymodule]
fn noiseroute_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RoutingCore>()?;
    Ok(())
}

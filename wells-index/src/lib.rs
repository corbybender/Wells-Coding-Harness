mod detect;
mod engine;
mod graph;
mod hash;
mod parse;
mod query;
mod store;

use pyo3::prelude::*;

#[pyclass]
pub struct IndexEngine {
    inner: engine::IndexEngine,
}

#[pymethods]
impl IndexEngine {
    #[new]
    fn new(workspace: &str) -> PyResult<Self> {
        let inner = engine::IndexEngine::new(workspace)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(IndexEngine { inner })
    }

    fn index(&mut self) -> PyResult<PyObject> {
        let stats = self
            .inner
            .index()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Python::with_gil(|py| {
            let dict = pyo3::types::PyDict::new_bound(py);
            dict.set_item("files_indexed", stats.files_indexed)?;
            dict.set_item("symbols_extracted", stats.symbols_extracted)?;
            dict.set_item("edges_extracted", stats.edges_extracted)?;
            dict.set_item("total_files", stats.total_files)?;
            dict.set_item("duration_ms", stats.duration_ms)?;
            Ok(dict.into())
        })
    }

    fn find_symbol(&self, name: &str) -> PyResult<PyObject> {
        let results = self
            .inner
            .find_symbol(name)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Python::with_gil(|py| {
            let list = pyo3::types::PyList::empty_bound(py);
            for result in results {
                let dict = pyo3::types::PyDict::new_bound(py);
                dict.set_item("file_path", result.file_path)?;
                dict.set_item("name", result.name)?;
                dict.set_item("kind", result.kind)?;
                dict.set_item("start_line", result.start_line)?;
                dict.set_item("end_line", result.end_line)?;
                list.append(dict)?;
            }
            Ok(list.into())
        })
    }

    fn find_references(&self, symbol: &str) -> PyResult<PyObject> {
        let results = self
            .inner
            .find_references(symbol)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Python::with_gil(|py| {
            let list = pyo3::types::PyList::empty_bound(py);
            for result in results {
                let dict = pyo3::types::PyDict::new_bound(py);
                dict.set_item("file_path", result.file_path)?;
                dict.set_item("name", result.name)?;
                dict.set_item("kind", result.kind)?;
                dict.set_item("start_line", result.start_line)?;
                dict.set_item("end_line", result.end_line)?;
                list.append(dict)?;
            }
            Ok(list.into())
        })
    }

    fn find_callers(&self, symbol: &str) -> PyResult<PyObject> {
        let results = self
            .inner
            .find_callers(symbol)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Python::with_gil(|py| {
            let list = pyo3::types::PyList::empty_bound(py);
            for result in results {
                let dict = pyo3::types::PyDict::new_bound(py);
                dict.set_item("file_path", result.file_path)?;
                dict.set_item("name", result.name)?;
                dict.set_item("kind", result.kind)?;
                dict.set_item("start_line", result.start_line)?;
                dict.set_item("end_line", result.end_line)?;
                list.append(dict)?;
            }
            Ok(list.into())
        })
    }

    fn search_symbols(&self, query: &str, limit: usize) -> PyResult<PyObject> {
        let results = self
            .inner
            .search_symbols(query, limit)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Python::with_gil(|py| {
            let list = pyo3::types::PyList::empty_bound(py);
            for result in results {
                let dict = pyo3::types::PyDict::new_bound(py);
                dict.set_item("file_path", result.file_path)?;
                dict.set_item("name", result.name)?;
                dict.set_item("kind", result.kind)?;
                dict.set_item("start_line", result.start_line)?;
                dict.set_item("end_line", result.end_line)?;
                list.append(dict)?;
            }
            Ok(list.into())
        })
    }

    fn list_in_file(&self, path: &str) -> PyResult<PyObject> {
        let results = self
            .inner
            .list_in_file(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Python::with_gil(|py| {
            let list = pyo3::types::PyList::empty_bound(py);
            for result in results {
                let dict = pyo3::types::PyDict::new_bound(py);
                dict.set_item("file_path", result.file_path)?;
                dict.set_item("name", result.name)?;
                dict.set_item("kind", result.kind)?;
                dict.set_item("start_line", result.start_line)?;
                dict.set_item("end_line", result.end_line)?;
                list.append(dict)?;
            }
            Ok(list.into())
        })
    }

    fn stats(&self) -> PyResult<PyObject> {
        let stats = self
            .inner
            .stats()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Python::with_gil(|py| {
            let dict = pyo3::types::PyDict::new_bound(py);
            dict.set_item("total_files", stats.total_files)?;
            dict.set_item("total_symbols", stats.total_symbols)?;
            dict.set_item("total_edges", stats.total_edges)?;
            dict.set_item("last_indexed_at", stats.last_indexed_at)?;
            Ok(dict.into())
        })
    }

    fn clear(&mut self) -> PyResult<()> {
        self.inner
            .clear()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(())
    }
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<IndexEngine>()?;
    Ok(())
}

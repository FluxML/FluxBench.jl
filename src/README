# FluxBench.jl

[![][buildkite-img]][buildkite-url] [![bench-img]][bench-url]

[buildkite-img]: https://badge.buildkite.com/560460043f33dc6a23b4bc7379e7dd120a2dc10b350d7021ca.svg
[buildkite-url]: https://buildkite.com/julialang/fluxbench-dot-jl

[bench-img]: https://img.shields.io/badge/Benchmarks-speed.fluxml.ai-blue
[bench-url]: https://speed.fluxml.ai

This is a repository that backs the results generated for https://speed.fluxml.ai

It is a collection of benchmarking runs for a subset of modeling done in the FluxML ecosystem and also serves as a means of tracking progress.

### Running Locally

To run the benchmarks locally:

* clone this repository
* `cd` in to the local copy via `cd FluxBench.jl`
* open Julia and call `] instantiate`

And finally:

```julia
julia> using FluxBench

julia> FluxBench.bench()
```

## Adding Benchmarks

To contribute benchmarks one needs to:
* add in the script(s) to the `src/packages` directory with the required dependencies and code needed to run the benchmarks
* include the benchmarks in the top level file `src/FluxBench.jl`
* call the benchmarks in the `bench` function located in file `src/bench.jl`

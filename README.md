# cmudb

This is, roughly, a Rust port of CMU's BusTub (https://github.com/cmu-db/bustub). It's a work in progress. 

## What's done:
- disk-backed pages for tables
- Adaptive replacement cache policy for pages
- B+ tree indexes with optimistic latch coupling
- An in-memory catalog
- Most executors described in the CMU course
- parser (using sqlparser crate)
- binder
- planner

## What's missing:
- optimizer
- mvcc
- support for more types, only a few basic types are supported (int, booleans, floats, timestamps)


## Things I would change if I have the time:
- Pages as typed views: Currently pages are fully decoded and encoded as we read from and write to them. A better approach is for pages act as typed views over raw bytes, which would save quite a bit of i/o work.
- Improved execution model: A volcano-style iteration model is used here, and notably there is no batching going on, so rows that the db system materializes are pulled 1 by 1.

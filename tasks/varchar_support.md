SUMMARY:
- Goal: Add VARCHAR support to the database via fixed-width-with-length-prefix storage. Each VARCHAR(N) column reserves N+2 bytes in the tuple payload (u16 length prefix + N reserved bytes for content, zero-padded).
- Context: Schema/Column code in `src/catalog/mod.rs` precomputes per-column offsets from `dtype.size()`, and the B+ tree expects a fixed `key_len`. Storing VARCHARs as fixed-width slots keeps both invariants intact, so the B+ tree, table heap, and buffer pool layers don't need changes. The trade-off is space waste (each value pads to its column's max), which is acceptable for a learning project.
- Outcome: TBD.

PLAN:
1. Add `DataType::VARCHAR(usize)` and `Value::VARCHAR(String)` (or `Vec<u8>`/`Box<str>` — pick one).
2. Drop `Copy` from `Value`. Fix the call sites that relied on it: `processor/plan.rs:eval_binary` and `processor/executor.rs:501,506,560`. Clone where needed.
3. Implement `DataType::size()` for VARCHAR as `n + 2`. This single change makes `Schema::new`'s offset math, `Schema::encode_tuple`'s NULL-padding path, and `Schema::get_value`'s slice math all work unchanged.
4. Implement `Value::encode` for VARCHAR: write u16 LE length, then exactly N bytes (truncate or zero-pad content to N). Implement `Value::decode`: read u16 length, then take first `length` bytes from the N-byte content region.
5. Add a `VARCHAR(n)` arm to `DataType::encode_from_tuple` (in `catalog/mod.rs`) and a matching arm to `TableIndex::encode_key_from_values` (in `catalog/index_schema.rs`). Sort-friendly encoding for the index key: emit the N content bytes left-as-is (no length prefix in the key — fixed-length bytewise lex order matches string lex order when content is zero-padded).
6. Define `MAX_VARCHAR_LEN` (e.g., 1024). Reject `VARCHAR(n)` with `n > MAX_VARCHAR_LEN` at `Schema::new`.
7. Compute `Schema::max_tuple_size()` and validate it at `Catalog::create_table` against `PAGE_SIZE - <table page header + slot dir budget>`. Reject schemas whose worst-case tuple won't fit on a page. This is the load-bearing "no spill" check.
8. Validate at encode time that an inbound `Value::VARCHAR(s)` has `s.len() <= n` for its column; surface as `CatalogError::TupleSchemaMismatch` (or a new variant if useful).
9. Tests:
   - Round-trip a tuple with VARCHAR columns through `encode_tuple` / `get_value`, including: empty string, max-length string, sub-max string, NULL.
   - Build a B+ tree index on a VARCHAR column; insert several keys, scan in order, confirm lex order.
   - Reject schema where summed tuple size exceeds page budget.
   - Reject `Value::VARCHAR(s)` whose `s.len()` exceeds the column's declared N.

LOGS:
- Date: 2026-05-09
  - What changed: Plan written. No code changes yet.
  - Why: Aligning on approach before implementation.
  - How validated: N/A.

ISSUES/ERRORS:
- (none yet)

NOTES / OPEN QUESTIONS:
- Length prefix width: u16 supports up to 65535; pairs naturally with `MAX_VARCHAR_LEN <= 65535`. If we ever raise the cap, widen to u32.
- Encoding choice: `String` vs `Vec<u8>` for the `Value::VARCHAR` payload. `String` is more ergonomic and gives us UTF-8 invariants for free; if we later want raw bytes (BLOB), introduce a separate variant.
- `Column.inlined` field: currently unused. Stays unused — VARCHAR is inlined under this design. Don't delete it; leaving the door open for a future TOAST-style overflow type isn't costly.
- Sort-friendly note: zero-padding works for tie-breaking only because `\0` < every printable byte. If we ever support binary varchars where `\0` is valid content, `"a\0"` vs `"a"` becomes ambiguous in the index. Fine for now.

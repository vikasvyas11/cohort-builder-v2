"""An uploaded model JSON carries raw SQL; it must not be able to read files, run
statements, change the sandbox or run forever. Everything here is deliberately small."""

import copy
import time

import duckdb
import pytest
import splink.blocking_rule_library as brl
import splink.comparison_library as cl

from modules import splink_runner as sr


def _settings(rule="l.\"first_name\" = r.\"first_name\"", level="\"first_name_l\" = \"first_name_r\""):
    return {
        "link_type": "dedupe_only",
        "blocking_rules_to_generate_predictions": [{"blocking_rule": rule, "sql_dialect": "duckdb"}],
        "comparisons": [{"output_column_name": "first_name",
                         "comparison_levels": [{"sql_condition": level}]}],
    }


# -- validator -----------------------------------------------------------------

@pytest.mark.parametrize("payload", [
    "1=1; DROP TABLE x",
    "l.\"a\" = r.\"a\" AND (SELECT count(*) FROM range(4000000000)) > 0",
    "l.\"a\" = r.\"a\" AND (SELECT 1) = 1",
    "l.\"a\" = r.\"a\" -- trailing comment",
    "l.\"a\" = r.\"a\" /* hidden */",
    "l.\"a\" IN (SELECT x FROM read_csv('/etc/passwd'))",
    "l.\"a\" = r.\"a\" AND len(read_text('secrets.toml')) > 0",
    "COPY (SELECT 1) TO 'out.csv'",
    "ATTACH 'x.db'",
    "l.\"a\" = r.\"a\" AND getenv('HOME') = 'x'",
    "l.\"a\" = r.\"a\" AND current_setting('memory_limit') = ''",
    "l.\"a\" = r.\"a\" UNION SELECT 1",
    "l.\"a\" = r.\"a\" AND parquet_scan('x') > 0",
    "x" * 5000,
])
def test_dangerous_sql_is_rejected_in_blocking_rules(payload):
    with pytest.raises(ValueError):
        sr.validate_model_sql(_settings(rule=payload))


def test_dangerous_sql_is_rejected_in_comparison_levels():
    with pytest.raises(ValueError):
        sr.validate_model_sql(_settings(level="(SELECT count(*) FROM range(10)) > 0"))


def test_keywords_inside_quoted_identifiers_and_strings_are_fine():
    # a column legitimately named "select" or a literal containing ';' must not trip the screen
    sr.validate_model_sql(_settings(rule='l."select" = r."select" AND l."note" <> \'a;b -- c\''))


def test_every_comparison_and_blocking_type_the_app_builds_passes_validation():
    comparisons = [
        cl.NameComparison("first_name"), cl.DateOfBirthComparison("dob", input_is_string=True),
        cl.EmailComparison("email"), cl.PostcodeComparison("postcode"), cl.ExactMatch("city"),
        cl.LevenshteinAtThresholds("surname", [1, 2]), cl.JaroWinklerAtThresholds("surname", [0.9, 0.7]),
        cl.JaroAtThresholds("surname", [0.9]), cl.ExactMatch("gender").configure(term_frequency_adjustments=True),
    ]
    rules = [brl.block_on("first_name"), brl.block_on("first_name", "surname"), brl.block_on("dob")]
    settings = {
        "comparisons": [c.create_comparison_dict("duckdb") for c in comparisons],
        "blocking_rules_to_generate_predictions": [r.create_blocking_rule_dict("duckdb") for r in rules],
    }
    sr.validate_model_sql(settings)


def test_app_exported_models_pass_validation(linkage_runs):
    for run in linkage_runs.values():
        sr.validate_model_sql(sr.reconstruct_model_json(run["settings_used"], run["model_params"]))


def test_run_linkage_from_json_refuses_a_malicious_model_before_touching_data(demo, linkage_runs):
    _, a, _ = demo
    model = copy.deepcopy(linkage_runs[("dedupe", "deterministic")]["settings_used"])
    model["blocking_rules_to_generate_predictions"] = [{
        "blocking_rule": 'l."first_name" = r."first_name" AND (SELECT count(*) FROM range(4000000000)) > 0',
        "sql_dialect": "duckdb"}]
    with pytest.raises(ValueError, match="not allowed"):
        sr.run_linkage_from_json(model, a, None, "dedupe", linkage_type="deterministic")


# -- sandbox (what happens if SQL gets past the screen) ----------------------------

@pytest.fixture
def con():
    db_api, watchdog = sr._guarded_db_api()
    yield db_api._con
    assert watchdog is None


def test_sandbox_blocks_file_access(con, tmp_path):
    target = tmp_path / "x.csv"
    target.write_text("a\n1\n")
    for sql in (f"SELECT * FROM read_csv('{target.as_posix()}')",
                f"COPY (SELECT 1) TO '{(tmp_path / 'out.csv').as_posix()}'",
                f"ATTACH '{(tmp_path / 'x.db').as_posix()}'"):
        with pytest.raises(duckdb.Error):
            con.execute(sql)


def test_sandbox_configuration_cannot_be_changed_by_sql(con):
    for sql in ("SET enable_external_access=true", "SET memory_limit='64GB'", "SET lock_configuration=false"):
        with pytest.raises(duckdb.Error):
            con.execute(sql)
    assert con.execute("SELECT current_setting('enable_external_access')").fetchone()[0] is False


def test_sandbox_memory_is_capped(con):
    limit = con.execute("SELECT current_setting('memory_limit')").fetchone()[0]
    assert limit.replace(" ", "").upper().startswith(("1.8", "1.9", "2.0", "1.86"))  # '2GB' -> ~1.9 GiB


def test_runaway_query_is_interrupted_by_the_watchdog():
    db_api, watchdog = sr._guarded_db_api(timeout_s=1)
    started = time.time()
    try:
        with pytest.raises(duckdb.InterruptException):
            db_api._con.execute("SELECT count(*) FROM range(100000000000) a, range(2) b").fetchall()
    finally:
        watchdog.cancel()
    assert time.time() - started < 30


def test_normal_runs_work_inside_the_sandbox(linkage_runs):
    assert all(run["n_edges"] > 0 for run in linkage_runs.values())


# -- generated SQL quoting ----------------------------------------------------------

def test_odd_column_names_cannot_break_generated_sql():
    import pandas as pd
    weird = 'name"; DROP TABLE x; --'
    df = pd.DataFrame({
        "unique_id_l": ["1"], "unique_id_r": ["2"], "source_dataset_l": "A", "source_dataset_r": "A",
        "match_probability": [0.9], "match_weight": [3.0],
        f"{weird}_l": ["a"], f"{weird}_r": ["a"],
    })
    out = sr.build_coverage_matrix(df, [weird])
    assert bool(out[f"covers_{weird}"].iloc[0]) is True
    assert sr._ident('a"b') == '"a""b"'


# -- single-CPU workaround ----------------------------------------------------------

def test_single_cpu_workaround_patches_only_when_needed_and_always_restores(monkeypatch):
    import multiprocessing
    real = multiprocessing.cpu_count
    monkeypatch.setattr(sr.os, "cpu_count", lambda: 1)
    with sr._at_least_two_cpus():
        assert multiprocessing.cpu_count() == 2
    assert multiprocessing.cpu_count is real

    monkeypatch.setattr(sr.os, "cpu_count", lambda: 8)
    with sr._at_least_two_cpus():
        assert multiprocessing.cpu_count is real       # untouched on a multi-CPU host

    monkeypatch.setattr(sr.os, "cpu_count", lambda: 1)
    with pytest.raises(RuntimeError):
        with sr._at_least_two_cpus():
            raise RuntimeError("boom")
    assert multiprocessing.cpu_count is real           # restored even on error

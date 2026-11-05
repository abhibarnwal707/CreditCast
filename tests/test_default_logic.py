"""Default / prepayment flag logic on small synthetic delinquency sequences.

Runs the real sql/build_panel.sql on hand-built loans/perf tables.
"""
import duckdb
import pytest

from creditcast.config import ROOT
from creditcast.data import run_sql_file


def months(start: int, n: int) -> list[int]:
    out, y, m = [], start // 100, start % 100
    for _ in range(n):
        out.append(y * 100 + m)
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def build_panel(loans: dict) -> "pd.DataFrame":
    """loans: {loan_id: [(period, upb, delq, zb_code), ...]}"""
    con = duckdb.connect()
    con.execute("CREATE TABLE loans (loan_id VARCHAR)")
    con.execute(
        "CREATE TABLE perf (loan_id VARCHAR, period INTEGER, qidx INTEGER, upb DOUBLE, delq INTEGER, "
        "age INTEGER, rem_months INTEGER, zb_code VARCHAR, rate DOUBLE, eltv DOUBLE, mod_flag VARCHAR)"
    )
    for lid, rows in loans.items():
        con.execute("INSERT INTO loans VALUES (?)", [lid])
        for i, (p, upb, delq, zb) in enumerate(rows):
            q = (p // 100) * 4 + ((p % 100) - 1) // 3
            con.execute("INSERT INTO perf VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                        [lid, p, q, upb, delq, i, 360 - i, zb, 5.0, None, None])
    run_sql_file(con, ROOT / "sql" / "build_panel.sql")
    return con.execute('SELECT * FROM panel ORDER BY loan_id, qidx').df()


def seq(start, delqs, zb_last=None, upb=100_000.0):
    ps = months(start, len(delqs))
    rows = []
    for i, (p, d) in enumerate(zip(ps, delqs)):
        last = i == len(delqs) - 1
        rows.append((p, 0.0 if (last and zb_last) else upb, d, zb_last if last else None))
    return rows


def test_always_current_never_defaults():
    # 2010-01 .. 2011-12 (8 quarters), always current, still alive at end of data
    df = build_panel({"A": seq(201001, [0] * 24)})
    assert df["default"].sum() == 0 and df["prepay"].sum() == 0
    # first row is 2010Q2 (quarter after the first observed quarter); last observed quarter is not a row
    assert df["qidx"].min() == 2010 * 4 + 1
    assert df["qidx"].max() == 2011 * 4 + 3


def test_sixty_days_is_not_default():
    df = build_panel({"A": seq(201001, [0, 0, 0, 1, 2, 2, 1, 0, 0, 0, 0, 0])})
    assert df["default"].sum() == 0


def test_ninety_days_is_default_in_that_quarter_and_absorbing():
    # reaches status 3 in 2010-07 => default in 2010Q3 (qidx 2010*4+2); later rows must not exist
    delqs = [0, 0, 0, 0, 0, 1, 2, 3, 3, 3, 0, 0, 0, 0, 0, 0]  # 2010-01..: index 7 is 2010-08 -> Q3
    df = build_panel({"A": seq(201001, delqs)})
    assert df["default"].sum() == 1
    d = df[df["default"] == 1].iloc[0]
    assert d["qidx"] == 2010 * 4 + 2
    assert df["qidx"].max() == d["qidx"]          # no rows after default (cure ignored)


def test_features_measured_at_start_of_quarter():
    # delinquent 60 days at the end of Q2 and 90 days in Q3 -> the default row shows start_delq = 2
    delqs = [0, 0, 0, 0, 1, 2, 3, 3, 3, 3]  # Jan..Oct 2010; Jun=2 (idx5), Jul=3 (idx6)
    df = build_panel({"A": seq(201001, delqs)})
    row = df[df["default"] == 1].iloc[0]
    assert row["qidx"] == 2010 * 4 + 2
    assert row["start_delq"] == 2


@pytest.mark.parametrize("zb", ["02", "03", "09", "15"])
def test_credit_event_zero_balance_is_default_even_without_90dpd(zb):
    delqs = [0] * 8
    df = build_panel({"A": seq(201001, delqs, zb_last=zb)})
    assert df["default"].sum() == 1 and df["prepay"].sum() == 0


def test_reo_acquisition_status_counts_as_default():
    # delq 99 stands for status "RA"
    df = build_panel({"A": seq(201001, [0, 0, 0, 0, 0, 0, 99, 99, 99, 99])})
    assert df["default"].sum() == 1


def test_prepayment_is_not_default():
    df = build_panel({"A": seq(201001, [0] * 8, zb_last="01")})
    assert df["default"].sum() == 0 and df["prepay"].sum() == 1


def test_other_exit_is_censored():
    df = build_panel({"A": seq(201001, [0] * 8, zb_last="96")})
    assert df["default"].sum() == 0 and df["prepay"].sum() == 0 and df["exit_other"].sum() == 1


def test_prepay_after_default_is_not_counted():
    delqs = [0, 0, 0, 3, 3, 3, 0, 0, 0, 0]
    df = build_panel({"A": seq(201001, delqs, zb_last="01")})
    assert df["default"].sum() == 1 and df["prepay"].sum() == 0


def test_no_duplicate_loan_quarters():
    df = build_panel({"A": seq(201001, [0] * 20), "B": seq(201004, [0, 0, 3, 3, 3, 3, 3, 3])})
    assert not df.duplicated(["loan_id", "qidx"]).any()

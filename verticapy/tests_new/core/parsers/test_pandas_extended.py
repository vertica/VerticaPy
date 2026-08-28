"""
Copyright  (c)  2018-2025 Open Text  or  one  of its
affiliates.  Licensed  under  the   Apache  License,
Version 2.0 (the  "License"); You  may  not use this
file except in compliance with the License.

You may obtain a copy of the License at:
http://www.apache.org/licenses/LICENSE-2.0

Unless  required  by applicable  law or  agreed to in
writing, software  distributed  under the  License is
distributed on an  "AS IS" BASIS,  WITHOUT WARRANTIES
OR CONDITIONS OF ANY KIND, either express or implied.
See the  License for the specific  language governing
permissions and limitations under the License.
"""

# Pytest
import pytest

# Standard Python Modules
import csv
import io
import time

from vertica_python.errors import CopyRejected

# Other Modules
import pandas

# VerticaPy
from verticapy import (
    drop,
)
from verticapy.connection import current_cursor
from verticapy.core.parsers.pandas import (
    read_pandas,
    DELIMITER,
    ENCLOSURE_CANDIDATES,
    ESCAPE_AS,
    RECORD_TERMINATOR,
    _pick_enclosure,
)
from verticapy.datasets import load_titanic


class TestPandasExtended:
    def test_read_pandas_abort_on_error(self, titanic_vd):
        """
        Tries to use read_pandas() to load a dataframe into a table
        that has the right column names, but the wrong column type
        for the data format. Asserts that abort_on_error behaves as
        expected.
        """
        pandas_df = titanic_vd.to_pandas()
        assert pandas_df.shape == (1234, 14)
        random_name = f"titanic_hack_{int(time.time())}"
        try:
            current_cursor().execute(
                f"create table public.{random_name} like"
                f" {titanic_vd} excluding projections"
            ).fetchall()
            current_cursor().execute(
                f'alter table public.{random_name} drop column "survived"'
            ).fetchall()
            current_cursor().execute(
                f'alter table public.{random_name} add column "survived" interval'
            ).fetchall()
            with pytest.raises(CopyRejected):
                read_pandas(
                    df=pandas_df,
                    name=random_name,
                    schema="public",
                    insert=True,
                    abort_on_error=True,
                )
        finally:
            current_cursor().execute(
                f"drop table if exists public.{random_name}"
            ).fetchall()


# Values that the intermediate CSV must carry through COPY untouched. The
# double-quote case is the regression test for the enclosure that read_pandas
# used to add by hand: it doubled an embedded quote, and a blanket
# .replace('""', "") then deleted both, so 'say "hi"' arrived as 'say hi'.
ROUND_TRIP_VALUES = {
    "comma": "a,b",
    "plain": "plain",
    "dquote": 'say "hi"',
    "backslash": "C:\\dir",
    "delimiter": "sep" + DELIMITER + "here",
    "terminator": "term" + RECORD_TERMINATOR + "here",
    "escape": "esc" + ESCAPE_AS + "here",
    "enclosure": "encl" + ENCLOSURE_CANDIDATES[0] + "here",
    "tab": "tab\tx",
    "empty": "",
    "null": None,
}


def _probe_frame():
    keys = list(ROUND_TRIP_VALUES)
    return pandas.DataFrame(
        {
            "k": keys,
            "txt": [ROUND_TRIP_VALUES[k] for k in keys],
            "allnull": [None] * len(keys),
            "num": list(range(len(keys))),
        }
    )


def _decode(text):
    """
    Splits the intermediate CSV into records and fields the way Vertica's
    DELIMITED parser does, undoing ESCAPE_AS but leaving any enclosure in
    place. A delimiter or record terminator preceded by ESCAPE_AS is data.
    """
    records, fields, current, i = [], [], "", 0
    while i < len(text):
        char = text[i]
        if char == ESCAPE_AS:
            current += text[i + 1]
            i += 2
        elif char == DELIMITER:
            fields.append(current)
            current, i = "", i + 1
        elif char == RECORD_TERMINATOR:
            fields.append(current)
            records.append(fields)
            fields, current, i = [], "", i + 1
        else:
            current += char
            i += 1
    if current or fields:
        fields.append(current)
        records.append(fields)
    return records


class TestReadPandasEncoding:
    """
    Covers the encoding of the intermediate CSV that read_pandas writes and
    COPY reads back.
    """

    def test_intermediate_csv_encoding(self):
        """
        Offline, no server: asserts the bytes to_csv produces. This is what
        pins the dialect, so that a pandas upgrade changing to_csv's escaping
        fails here rather than as silent data corruption in a live ingest.
        """
        df = _probe_frame()
        enclosed_by = _pick_enclosure(df, ["k", "txt"])
        # The first candidate occurs in the data, so the next must be chosen.
        assert enclosed_by == ENCLOSURE_CANDIDATES[1]

        tmp_df = df.copy()
        for c in ("k", "txt"):
            tmp_df[c] = enclosed_by + tmp_df[c].str.slice() + enclosed_by
        tmp_df["allnull"] = ""
        buf = io.StringIO()
        tmp_df.to_csv(
            buf,
            index=False,
            quoting=csv.QUOTE_NONE,
            quotechar=None,
            escapechar=ESCAPE_AS,
            sep=DELIMITER,
            lineterminator=RECORD_TERMINATOR,
        )
        records = _decode(buf.getvalue())
        # The header, plus one record per value - the escaped delimiters and
        # record terminators in the data must not have split anything.
        assert len(records) == len(ROUND_TRIP_VALUES) + 1
        assert records[0] == ["k", "txt", "allnull", "num"]
        fields = {record[0][1:-1]: record[1] for record in records[1:]}

        for key, value in ROUND_TRIP_VALUES.items():
            if value is None:
                # A NA is written as an empty, unenclosed field so that
                # COPY's NULL '' matches it.
                assert fields[key] == "", key
            else:
                assert fields[key] == enclosed_by + value + enclosed_by, key

        # The old scheme's fingerprint: it is the doubling of the quote, and
        # the blanket replace that removed it, that lost the quotes.
        assert '""' not in buf.getvalue()

    def test_pick_enclosure_exhausted(self):
        """
        Data containing every candidate leaves nothing to enclose with.
        """
        df = pandas.DataFrame({"txt": ["".join(ENCLOSURE_CANDIDATES)]})
        assert _pick_enclosure(df, ["txt"]) is None

    def test_pick_enclosure_skips_column_names(self):
        """
        A candidate occurring in a column name is rejected too - the header
        row goes through the same encoding as the data.
        """
        df = pandas.DataFrame({"a" + ENCLOSURE_CANDIDATES[0]: ["x"]})
        assert _pick_enclosure(df, []) == ENCLOSURE_CANDIDATES[1]

    def test_no_enclosure_available_suggests_dtype(self):
        """
        When no enclosure is available the type guess is what breaks, so the
        error has to point at the parameter that skips it.
        """
        df = pandas.DataFrame({"txt": ["".join(ENCLOSURE_CANDIDATES)]})
        with pytest.raises(ValueError, match="dtype"):
            read_pandas(df=df, name=f"never_{int(time.time())}", schema="public")

    @pytest.mark.parametrize("mode", ["dtype", "insert"])
    def test_ingest_without_enclosure(self, mode):
        """
        Supplying dtype, or inserting into an existing relation, skips the
        type guess - so the fields need no enclosing and a value holding
        every candidate control character still round trips.
        """
        every = "".join(ENCLOSURE_CANDIDATES)
        df = pandas.DataFrame(
            {"k": ["all", "quote"], "txt": ["x" + every + "y", 'say "hi"']}
        )
        dtype = {"k": "varchar(20)", "txt": "varchar(60)"}
        random_name = f"read_pandas_noenc_{int(time.time())}"
        try:
            if mode == "insert":
                read_pandas(df=df, name=random_name, schema="public", dtype=dtype)
                current_cursor().execute(
                    f"delete from public.{random_name}"
                ).fetchall()
                current_cursor().execute("commit").fetchall()
                vdf = read_pandas(
                    df=df,
                    name=random_name,
                    schema="public",
                    insert=True,
                    abort_on_error=True,
                )
            else:
                vdf = read_pandas(
                    df=df, name=random_name, schema="public", dtype=dtype
                )
            got = vdf.to_pandas()
            assert len(got) == 2
            assert got[got["k"] == "all"]["txt"].iloc[0] == "x" + every + "y"
            assert got[got["k"] == "quote"]["txt"].iloc[0] == 'say "hi"'
        finally:
            drop(f"public.{random_name}", method="table")

    @pytest.mark.parametrize("insert", [False, True])
    def test_round_trip_special_characters(self, insert):
        """
        Live round trip of the values above, on both the read_csv path
        (insert=False) and the COPY path (insert=True). The row count is
        asserted as well as the values: an encoding Vertica rejects drops
        the offending row rather than mangling it.
        """
        df = _probe_frame()
        random_name = f"read_pandas_enc_{int(time.time())}"
        try:
            if insert:
                # insert=True needs the relation to exist already; empty it so
                # that only the COPY path's own rows are asserted on.
                read_pandas(df=df, name=random_name, schema="public")
                current_cursor().execute(
                    f"delete from public.{random_name}"
                ).fetchall()
                current_cursor().execute("commit").fetchall()
                vdf = read_pandas(
                    df=df,
                    name=random_name,
                    schema="public",
                    insert=True,
                    abort_on_error=True,
                )
            else:
                vdf = read_pandas(df=df, name=random_name, schema="public")
            got = vdf.to_pandas()

            assert len(got) == len(ROUND_TRIP_VALUES)
            assert got["allnull"].isna().all()
            for key, expected in ROUND_TRIP_VALUES.items():
                row = got[got["k"] == key]
                assert len(row) == 1, key
                actual = row["txt"].iloc[0]
                if expected is None:
                    assert pandas.isna(actual), key
                else:
                    assert actual == expected, key
        finally:
            drop(f"public.{random_name}", method="table")

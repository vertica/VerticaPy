#!/usr/bin/env python3
"""
Copyright  (c)  2018-2024 Open Text  or  one  of its
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
"""This script runs has helpful functions for the pipeline. """

# required packages
import re
from verticapy._utils._sql._sys import _executeSQL


def required_keywords(yaml: dict, keywords: list) -> bool:
    # if false raise keyerror
    for keyword in keywords:
        if keyword not in yaml.keys():
            raise KeyError(f"Missing keyword(s): '{keyword}'")
    return True


def execute_and_add(sql: str) -> str:
    _executeSQL(sql)
    return sql + "\n"


def remove_comments(input_string):
    pattern = r"/\*.*?\*/"
    result = re.sub(pattern, "", input_string, flags=re.DOTALL)
    return result


def to_sql(sql):
    even = True
    string = ""
    for c in sql:
        if even and c == "'":
            string += "' || QUOTE_LITERAL('"
            even = False
        elif not even and c == "'":
            string += "') || '"
            even = True
        else:
            string += c
    string = string[:-1] + "';"
    return string


def setup():
    # CREATE drop_pipeline
    _executeSQL(
        """CREATE OR REPLACE PROCEDURE drop_pipeline(schem VARCHAR, recipe VARCHAR) LANGUAGE PLvSQL AS $$
            DECLARE
                v_name VARCHAR;
            BEGIN
                FOR v_name IN QUERY SELECT trigger_name FROM STORED_PROC_TRIGGERS 
                WHERE trigger_name LIKE recipe || '_%' AND schema_name LIKE schem
                LOOP
                    EXECUTE 'DROP TRIGGER ' || v_name;
                END LOOP;
                FOR v_name IN QUERY SELECT schedule_name FROM USER_SCHEDULES 
                WHERE schedule_name LIKE recipe || '_%' AND schema_name LIKE schem
                LOOP
                    EXECUTE 'DROP SCHEDULE ' || v_name;
                END LOOP;
                FOR v_name IN QUERY SELECT model_name FROM MODELS 
                WHERE model_name LIKE recipe || '_%' AND schema_name LIKE schem
                LOOP
                    EXECUTE 'DROP MODEL ' || v_name;
                END LOOP;
                FOR v_name IN QUERY SELECT table_name FROM v_catalog.views 
                WHERE table_name LIKE recipe || '_%' AND table_schema LIKE schem
                LOOP
                    EXECUTE 'DROP VIEW ' || v_name ;
                END LOOP;
                FOR v_name IN QUERY SELECT table_name FROM v_catalog.tables 
                WHERE table_name LIKE recipe || '_%' AND table_schema LIKE schem
                LOOP
                    EXECUTE 'DROP TABLE ' || v_name ;
                END LOOP;
                FOR v_name IN QUERY SELECT name FROM DATA_LOADERS 
                WHERE name LIKE recipe || '_%' AND schemaname LIKE schem
                LOOP
                    EXECUTE 'DROP DATA LOADER ' || v_name;
                END LOOP;
                FOR v_name IN QUERY SELECT procedure_name FROM v_catalog.user_procedures 
                WHERE procedure_name LIKE recipe || '_%' AND schema_name LIKE schem
                LOOP
                    EXECUTE 'DROP PROCEDURE ' || v_name || '()';
                END LOOP;
            END;
            $$;"""
    )


if __name__ == "__main__":
    setup()
    _executeSQL("CALL drop_pipeline('public', 'recipeIngestion');")
